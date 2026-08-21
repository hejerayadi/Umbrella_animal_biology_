from data_gbif import documents


print("=" * 60)
print("VERIFICATION DES DONNEES GBIF")
print("=" * 60)

print(f"\nNombre total d'observations : {len(documents)}")


# Regrouper les observations par espèce
species_data = {}

for doc in documents:

    species = doc["species"]

    if species not in species_data:
        species_data[species] = []

    species_data[species].append(doc)


# Affichage
for species, observations in species_data.items():

    print("\n" + "-" * 60)

    print(f"Espèce : {species}")

    print(
        f"Nombre d'observations : "
        f"{len(observations)}"
    )

    print("\nQuelques observations :")

    for obs in observations[:5]:

        print(
            f"  Date : {obs['event_date']}"
            f" | Position : "
            f"{obs['latitude']}, "
            f"{obs['longitude']}"
        )


print("\n" + "=" * 60)
print("FIN DE LA VERIFICATION")
print("=" * 60)