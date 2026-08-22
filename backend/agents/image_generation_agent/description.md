Generates 2D scientific images with FLUX.2-pro from the species, protein and trait context collected by the other agents. It needs a subject to draw — a species or a protein identifier, either one on its own is enough.

Example questions:
- "Draw a woolly mammoth"
- "Draw the structural traits of human insulin"
- "Create a 2D illustration of woolly mammoth morphological traits"
- "Generate a scientific image for the species described in context"

This agent analyzes nothing itself and predicts no structures — it is the rendering step. It requests trait interpretation from the Trait Discovery Agent when traits are missing, organizes the returned traits into visualization sections, builds a prompt, and calls FLUX.2-pro.

For 3D protein structure prediction (AlphaFold / ESMFold), see the separate 3D Protein Structure Agent.
