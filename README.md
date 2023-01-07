# Preheat2022

In this repository there are the key codes and modules for producing the results of my [preheat paper](link to be add).

The code implement following functions:

1. [X] [A general framework](./field_util.py) to set up a coordinate system. the convertion between (RA, Dec, Redshift) in the celestial coordinates system and (x, y, z) in a comoving system.
2. [X] Manipulating the fields with [Python code](./field_util.py). Equip raw field arrays with coordinates, clip these fields by specifying the coordinates, extract the values inside a sphere, etc.
3. [X] [Dealing with dark matter simulation outputs](./FGPA.py). "Paint" the particle data as fields, and [full procedure](./handle_dm_sim.ipynb) to apply FGPA.
4. [X] Running mock Wiener Filter like CLAMATO. [Prepering the data](./wiener_filter_input.ipynb), [feeding it to dachshund](./wiener_filter_mock.ipynb).
5. [X] [LSS stuff](./identify_pc_member.ipynb): idenfitying protocluster members, calculating Virial mass, and further follow up with Colossus.
6. [ ] Plots & visualization
    - [ ] Transmission-density relation shown with 2d histogram
    - [ ] Pixel data shown with 1d histrogram
    - [X] [Sliceplot](./sliceplot_mockrec.ipynb) for sanity check / visual inspect
    - [X] [Histrograms](./preheat_detection) which illustrate the preheating signal
    - [X] [A field visualization](./vis_util_o3d.py) based on open3d (native, powerful but many wheels to be invented)
    - [X] [A field visualization](./interactive) based on x3dom.js (beautiful, supported by aas journals, but not optimal if you are not familar with front-end stuff. Also check [this notebook](./protocluster_viz.ipynb) about generating the visualization)
7. [X] [A void prober](./VoidProber.py) and [an overdensity prober](ProtoClusterProber.py), which serve as the by-products of some literature review.
