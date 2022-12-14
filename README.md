# Preheat2022

In this repository there are the key codes and modules for producing the results of my [preheat paper](link to be add).

The code implement following functions:

1. A general framework to set up a coordinate system. the convertion between (RA, Dec, Redshift) in the celestial coordinates system and (x, y, z) in a comoving system.
2. Manipulating the fields with Python code. Equip raw field arrays with coordinates, clip these fields by specifying the coordinates, extract the values inside a sphere, etc.
3. Dealing with dark matter simulation outputs. "Paint" the particle data as fields, and full procedure to apply FGPA.
4. Plots & visualizationthe
   - transmission-density relation shown with 2d histogram
   - Pixel data shown with 1d histrogram
   - Sliceplot for sanity check / visual inspect
   - A field visualization based on open3d (native, powerful but many wheels to be invented)
   - A field visualization based on x3dom.js (beautiful, supported by aas, but not optimal if you are not familar with front-end stuff.)
5. A void prober and an overdensity prober, which serve as the by-products of some literature review.
