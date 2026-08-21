import streamlit as st
import math

import folium
from streamlit_folium import st_folium

from predict import (
    process_agent_query,
    predict_next_route
)
from enrich_data import enrich_documents
from data_gbif import documents
from gemini_api import answer_migration_question


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BioMigrate AI",
    page_icon="🌿",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Bonjour ! 🦅 Posez-moi une question "
                "sur la migration d'un animal."
            )
        }
    ]


if "current_routes" not in st.session_state:

    st.session_state.current_routes = []


if "current_species" not in st.session_state:

    st.session_state.current_species = "Globe Global"


if "predicted_route" not in st.session_state:

    st.session_state.predicted_route = None


# Permet d'éviter de relancer plusieurs fois
# exactement la même prédiction
if "last_clicked_position" not in st.session_state:

    st.session_state.last_clicked_position = None


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }

    .map-title {
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LAYOUT
# ============================================================

col_map, col_chat = st.columns(
    [1.5, 1],
    gap="large"
)


# ============================================================
# CHATBOT — DROITE
# ============================================================

with col_chat:

    st.subheader("💬 Agent Conversationnel")

    chat_container = st.container(
        height=500
    )


    # ========================================================
    # HISTORIQUE
    # ========================================================

    for message in st.session_state.messages:

        with chat_container.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )


    # ========================================================
    # QUESTION
    # ========================================================

    if user_input := st.chat_input(
        "Exemple : Où migre la cigogne blanche ?"
    ):

        # ----------------------------------------------------
        # MESSAGE UTILISATEUR
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )


        # ----------------------------------------------------
        # AGENT
        # ----------------------------------------------------

        try:

            species, routes = process_agent_query(
                user_input
            )
            enriched_documents = enrich_documents(
                   documents,
                 limit=5
            )

            response_text = answer_migration_question(
                 user_input,
                enriched_documents,
                st.session_state.predicted_route
            )

                        # =================================================
            # ROUTES TROUVÉES
            # =================================================

            if routes:

                response_text = (
                    f"✅ **Espèce :** "
                    f"{species.capitalize()}\n\n"

                    f"🗺️ **{len(routes)} observation(s) "
                    f"de migration trouvée(s).**\n\n"

                    f"🌍 Les trajectoires migratoires "
                    f"sont affichées sur la carte.\n\n"

                    f"🦅 Cliquez sur un point de départ "
                    f"pour prédire la prochaine position."
                )

                # Enrichissement GBIF + météo
                enriched_documents = enrich_documents(
                    documents,
                    limit=5
                )

                # Analyse Gemini
                gemini_response = answer_migration_question(
                    user_input,
                    enriched_documents,
                    st.session_state.predicted_route
                )

                # Ajouter la réponse Gemini
                response_text += "\n\n🤖 **Analyse :**\n\n"
                response_text += gemini_response

                st.session_state.current_routes = routes

                st.session_state.current_species = species

                # Nouvelle question :
                # on supprime l'ancienne prédiction

                st.session_state.predicted_route = None

                # On réinitialise le dernier clic

                st.session_state.last_clicked_position = None

            else:

                response_text = (
                    "⚠️ Aucune observation migratoire "
                    "avec coordonnées GPS n'a été trouvée."
                )

                st.session_state.current_routes = []

                st.session_state.predicted_route = None

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response_text
                }
            )

        except Exception as e:

            response_text = (
                "❌ Une erreur est survenue "
                "pendant le traitement."
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response_text
                }
            )

            st.error(
                f"Erreur : {str(e)}"
            )

        # Rechargement pour afficher
        # les nouvelles routes

        st.rerun()


# ============================================================
# CARTE — GAUCHE
# ============================================================

with col_map:

    st.markdown(
        f"""
        <div class="map-title">
            🌍 Migration de :
            {st.session_state.current_species.capitalize()}
        </div>
        """,
        unsafe_allow_html=True
    )


    routes = st.session_state.current_routes


    # ========================================================
    # CENTRE INITIAL
    # ========================================================

    map_center = [
        20,
        0
    ]

    zoom_start = 2


    # ========================================================
    # SI DES ROUTES EXISTENT
    # ========================================================

    if routes:

        first_route = routes[0]

        try:

            map_center = [
                float(first_route["start_lat"]),
                float(first_route["start_lon"])
            ]

            zoom_start = 3

        except Exception:

            map_center = [
                20,
                0
            ]

            zoom_start = 2


    # ========================================================
    # CREATION DE LA CARTE
    # ========================================================

    m = folium.Map(

        location=map_center,

        zoom_start=zoom_start,

        tiles="CartoDB dark_matter",

        control_scale=True

    )


    # ========================================================
    # ROUTES MIGRATOIRES EXISTANTES
    # ========================================================

    all_points = []


    for index, route in enumerate(routes):


        # ====================================================
        # COORDONNEES
        # ====================================================

        try:

            start_lat = float(
                route["start_lat"]
            )

            start_lon = float(
                route["start_lon"]
            )

            end_lat = float(
                route["end_lat"]
            )

            end_lon = float(
                route["end_lon"]
            )

        except (KeyError, TypeError, ValueError):

            # Si la route ne possède pas de destination,
            # on l'ignore pour l'affichage de la trajectoire.

            continue


        # ====================================================
        # STOCKAGE DES POINTS
        # ====================================================

        all_points.append(
            [
                start_lat,
                start_lon
            ]
        )

        all_points.append(
            [
                end_lat,
                end_lon
            ]
        )


        # ====================================================
        # POINT DE DEPART
        # ====================================================

        folium.Marker(

            location=[
                start_lat,
                start_lon
            ],

            popup=(
                f"<b>🦅 Position observée</b><br>"
                f"Espèce : "
                f"{route.get('species', st.session_state.current_species)}"
                f"<br>"
                f"Latitude : {start_lat:.4f}<br>"
                f"Longitude : {start_lon:.4f}<br>"
                f"<br>"
                f"<b>👆 Cliquez sur ce point "
                f"pour prédire la prochaine position.</b>"
            ),

            tooltip=(
                "🦅 Cliquez pour prédire "
                "la prochaine position"
            ),

            icon=folium.Icon(
                color="green",
                icon="play",
                prefix="fa"
            )

        ).add_to(m)


        # ====================================================
        # DESTINATION EXISTANTE
        # ====================================================

        folium.Marker(

            location=[
                end_lat,
                end_lon
            ],

            popup=(
                f"<b>🔴 Destination observée</b><br>"
                f"Espèce : "
                f"{route.get('species', st.session_state.current_species)}"
            ),

            tooltip="🔴 Destination",

            icon=folium.Icon(
                color="red",
                icon="flag",
                prefix="fa"
            )

        ).add_to(m)


        # ====================================================
        # GENERATION DE LA TRAJECTOIRE COURBEE
        # ====================================================

        points = []

        number_of_points = 60


        for i in range(
            number_of_points + 1
        ):

            t = (
                i /
                number_of_points
            )


            # Latitude

            lat = (
                start_lat
                +
                (
                    end_lat
                    -
                    start_lat
                ) * t
            )


            # Longitude

            lon = (
                start_lon
                +
                (
                    end_lon
                    -
                    start_lon
                ) * t
            )


            # Courbure

            curve = (
                math.sin(
                    math.pi * t
                )
                *
                8
            )


            lat += curve


            points.append(
                [
                    lat,
                    lon
                ]
            )


        # ====================================================
        # TRAJECTOIRE EXISTANTE
        # ====================================================

        folium.PolyLine(

            points,

            color="#35e89a",

            weight=5,

            opacity=0.9,

            tooltip=(
                f"Migration observée "
                f"{index + 1}"
            )

        ).add_to(m)


        # ====================================================
        # PETITS POINTS SUR LA TRAJECTOIRE
        # ====================================================

        for i in range(
            0,
            len(points),
            10
        ):

            folium.CircleMarker(

                location=points[i],

                radius=3,

                color="#ffffff",

                fill=True,

                fill_color="#35e89a",

                fill_opacity=1

            ).add_to(m)


    # ========================================================
    # ROUTE PREDITE PAR LE MODELE ML
    # ========================================================

    prediction = (
        st.session_state.predicted_route
    )


    if prediction:

        predicted_start_lat = float(
            prediction["start_lat"]
        )

        predicted_start_lon = float(
            prediction["start_lon"]
        )

        predicted_lat = float(
            prediction["predicted_lat"]
        )

        predicted_lon = float(
            prediction["predicted_lon"]
        )


        # ====================================================
        # GENERATION DE LA ROUTE PREDITE
        # ====================================================

        predicted_points = []

        number_of_points = 60


        for i in range(
            number_of_points + 1
        ):

            t = (
                i /
                number_of_points
            )


            lat = (
                predicted_start_lat
                +
                (
                    predicted_lat
                    -
                    predicted_start_lat
                ) * t
            )


            lon = (
                predicted_start_lon
                +
                (
                    predicted_lon
                    -
                    predicted_start_lon
                ) * t
            )


            # Courbure

            curve = (
                math.sin(
                    math.pi * t
                )
                *
                5
            )


            lat += curve


            predicted_points.append(
                [
                    lat,
                    lon
                ]
            )


        # ====================================================
        # ROUTE ORANGE
        # ====================================================

        folium.PolyLine(

            predicted_points,

            color="#f29107",

            weight=6,

            opacity=0.95,

            dash_array="10, 10",

            tooltip=(
                "🔮 Route migratoire prédite "
                "par le modèle ML"
            )

        ).add_to(m)


        # ====================================================
        # POSITION ACTUELLE
        # ====================================================

        folium.CircleMarker(

            location=[
                predicted_start_lat,
                predicted_start_lon
            ],

            radius=8,

            color="#f29107",

            fill=True,

            fill_color="#f29107",

            fill_opacity=1,

            popup=(
                "<b>📍 Position actuelle</b><br>"
                f"Latitude : "
                f"{predicted_start_lat:.4f}<br>"
                f"Longitude : "
                f"{predicted_start_lon:.4f}"
            )

        ).add_to(m)


        # ====================================================
        # DESTINATION PREDITE
        # ====================================================

        folium.Marker(

            location=[
                predicted_lat,
                predicted_lon
            ],

            popup=(
                "<b>🔮 Prochaine position prédite</b><br>"
                f"Latitude : "
                f"{predicted_lat:.4f}<br>"
                f"Longitude : "
                f"{predicted_lon:.4f}<br>"
                f"<br>"
                f"<b>Modèle : Random Forest</b>"
            ),

            tooltip=(
                "🔮 Prochaine position prédite"
            ),

            icon=folium.Icon(

                color="orange",

                icon="flag",

                prefix="fa"

            )

        ).add_to(m)


        # Ajouter les points prédits
        # pour ajuster la carte

        all_points.append(
            [
                predicted_start_lat,
                predicted_start_lon
            ]
        )

        all_points.append(
            [
                predicted_lat,
                predicted_lon
            ]
        )


    # ========================================================
    # AJUSTEMENT AUTOMATIQUE DE LA CARTE
    # ========================================================

    if len(all_points) >= 2:

        m.fit_bounds(
            all_points,
            padding=(30, 30)
        )


    # ========================================================
    # LEGENDE
    # ========================================================

    legend_html = """

    <div style="
        position: fixed;
        bottom: 25px;
        left: 25px;
        z-index: 9999;

        background: rgba(0,0,0,0.82);

        padding: 12px 16px;

        border-radius: 10px;

        color: white;

        font-family: Arial;

        font-size: 13px;

        box-shadow:
            0 3px 12px
            rgba(0,0,0,0.35);
    ">

        <b>🌍 Migration</b>

        <br><br>

        <span style="color:#35e89a;">
            ●
        </span>

        Position / trajet observé

        <br>

        <span style="color:#ff4b55;">
            ●
        </span>

        Destination observée

        <br>

        <span style="color:#35e89a;">
            ━━━━━
        </span>

        Trajectoire existante

        <br>

        <span style="color:#f29107;">
            ━ ━ ━
        </span>

        Route prédite par ML 🔮

    </div>

    """


    m.get_root().html.add_child(
        folium.Element(
            legend_html
        )
    )


    # ========================================================
    # AFFICHAGE DE LA CARTE
    # ========================================================

    map_data = st_folium(

        m,

        width=None,

        height=620,

        returned_objects=[
            "last_object_clicked"
        ]

    )


    # ========================================================
    # DETECTION DU CLIC
    # ========================================================

    if map_data:

        clicked = map_data.get(
            "last_object_clicked"
        )


        if clicked:

            clicked_lat = clicked.get(
                "lat"
            )

            clicked_lon = clicked.get(
                "lng"
            )


            if (
                clicked_lat is not None
                and
                clicked_lon is not None
            ):

                clicked_position = (
                    round(float(clicked_lat), 6),
                    round(float(clicked_lon), 6)
                )


                # =================================================
                # EVITER UNE REPETITION DU MEME CLIC
                # =================================================

                if (
                    st.session_state.last_clicked_position
                    != clicked_position
                ):

                    st.session_state.last_clicked_position = (
                        clicked_position
                    )


                    # =============================================
                    # PREDICTION ML
                    # =============================================

                    try:

                        prediction = predict_next_route(

                            species=(
                                st.session_state.current_species
                            ),

                            current_lat=float(
                                clicked_lat
                            ),

                            current_lon=float(
                                clicked_lon
                            )

                        )


                        # Sauvegarde de la prédiction

                        st.session_state.predicted_route = (
                            prediction
                        )


                        # =========================================
                        # MESSAGE DANS LE CHAT
                        # =========================================

                        response_text = (

                            f"📍 **Position actuelle :** "
                            f"{float(clicked_lat):.4f}, "
                            f"{float(clicked_lon):.4f}\n\n"

                            f"🔮 **Prochaine position prédite :** "
                            f"{prediction['predicted_lat']:.4f}, "
                            f"{prediction['predicted_lon']:.4f}\n\n"

                            f"🤖 **Modèle ML :** "
                            f"Random Forest\n\n"

                            f"🛣️ **La route migratoire "
                            f"prédite est affichée en orange "
                            f"sur la carte.**"
                        )


                        st.session_state.messages.append(

                            {
                                "role": "assistant",
                                "content": response_text
                            }

                        )


                        # =========================================
                        # RECHARGER LA PAGE
                        # =========================================

                        st.rerun()


                    except Exception as e:

                        st.error(
                            "❌ Erreur pendant la prédiction : "
                            f"{str(e)}"
                        )