# Neural-3D-Reconstruction-with-Material-Priors

As of late, neural implicit surface reconstruction methods have gained popularity in the ever-growing field of multiview 3D reconstruction. Many preceding
works have managed to produce remarkable reconstruction results, especially for complex indoor scenes. Furthermore, it has been established that priors are essential for neural implicit functions to reconstruct high quality surfaces from input RGB images. Nonetheless, concurrent reconstruction methods make use of priors
that require large-scale pre-training and merely provide geometric and colour cues, whilst ignoring the underlying material properties. 

Motivated by recent advancements in direct scene material prediction from single-view images, we explore the utility that these cues provide for improving neural implicit surface reconstruction. Specifically, we incorporate the material priors predicted from a generative model as regularizers to optimize signed distance fields via inverse rendering for the reconstruction of intricate indoor scenes. With this, we are able to recover 3D neural fields of surface normals, light visibility, albedo and Bidirectional Reflectance Distribution Functions (BRDFs) and jointly refine the geometry simultaneously.

Below is a visual summary of our proposed neural reconstruction method, with the help of material priors.

![Intro to Method - Visual Summary](https://github.com/jngsa/Neural-3D-Reconstruction-with-Material-Priors/blob/9e9550b172930a34b390d5d2b11c8b4a6c55af78/img/intro-method.png)

1. **Material Prior Prediction**

For the purpose of predicting underlying material properties, we make use of [Intrinsic Image Diffusion](https://github.com/Peter-Kocsis/IntrinsicImageDiffusion), or IID.
IID is a pre-trained Stable Diffusion model for decomposing single-view images into geometry, material, and lighting conditions.
In our method, only the material prediction part of IID contributes to our prior prediction.
These material properties from IID are then seen as material ground truths.
We therefore want to minimize the L1 loss between estimated materials and ground truth materials during optimization.


2. **Optimization via Inverse Rendering**

For our pre-trained model, we opt for a pre-trained INGP NeuS model trained on the scene for 500k epochs.
This pre-trained model contributes its SDF and colour network.
We use a pre-trained INGP NeuS model because carrying out inverse rendering from scratch causes instability during optimization, thus it needs an initialization, provided by the pre-trained model. 
From the pre-trained INGP NeuS model, we use sampled depths to compute the expected depth for surface point computations.
Directly from its SDF, we can extract its normals via gradient computation.
Similar to our NeRFactor implementations, we maintain a BRDF MLP for metallic and roughness, and an Albedo MLP for albedo prediction.
Light visibility computation works the same way as explained earlier, just without TSDF Fusion, as we can extract a high quality mesh directly from INGP NeuS.
All of these components are fed into the rendering equation for re-rendering of the scene.

