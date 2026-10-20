# Neural-3D-Reconstruction-with-Material-Priors

As of late, neural implicit surface reconstruction methods have gained popularity in the ever-growing field of multiview 3D reconstruction. Many preceding
works have managed to produce remarkable reconstruction results, especially for complex indoor scenes. Furthermore, it has been established that priors are essential for neural implicit functions to reconstruct high quality surfaces from input RGB images. Nonetheless, concurrent reconstruction methods make use of priors
that require large-scale pre-training and merely provide geometric and colour cues, whilst ignoring the underlying material properties. 

Motivated by recent advancements in direct scene material prediction from single-view images, we explore the utility that these cues provide for improving neural implicit surface reconstruction. Specifically, we incorporate the material priors predicted from a generative model as regularizers to optimize signed distance fields via inverse rendering for the reconstruction of intricate indoor scenes. With this, we are able to recover 3D neural fields of surface normals, light visibility, albedo and Bidirectional Reflectance Distribution Functions (BRDFs) and jointly refine the geometry simultaneously.

Below is a visual summary of our proposed neural reconstruction method, with the help of material priors.

![Intro to Method - Visual Summary](https://github.com/jngsa/Neural-3D-Reconstruction-with-Material-Priors/blob/9e9550b172930a34b390d5d2b11c8b4a6c55af78/img/intro-method.png)
