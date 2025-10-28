![GitHub Tag](https://img.shields.io/github/v/tag/j-haacker/aris_lite)
![GitHub top language](https://img.shields.io/github/languages/top/j-haacker/aris_lite)
![GitHub License](https://img.shields.io/github/license/j-haacker/aris_lite)

# ARIS_lite

ARIS models plant growth based on environmental parameter. The model
draws on the references at the bottom.

## 🌱 state

The model has been validated against the orignal ARIS model. This is not
a stable software - future changes may break your work, but I will try
not to.

## 🪛 usage

1. `python water_budget.py -m snow 2019 2020 2021 2022 2023`
2. `python phenology.py`
3. `python water_budget.py -m soil`
4. `python yield_expectation.py`

## ✨ features

- calculate water up-take coefficients ("Kc factors") for Winter Wheat,
    Spring Barley, Maize, and Grassland based on the daily air surface
    temperature
- calculate soil water content and evapotranspiration
- compute daily crop-specific stress index based on maximum surface air
    temperature and soil water saturation
- estimate yield expectations based on stress index

## 📑 API documentation

<https://aris-lite.readthedocs.io>

## 🔗 dependencies

- dask, numpy, pandas, snowmaus, xarray, zarr
- meteorological data
- soil water capacity data

## 🐛 known issues

- hard-coded observable names, e.g. "max_air_temp"

## 💸 funding

The implementation of ARIS_lite in Python, this repository, is funded by
the Austrian Research Promotion Agency (FFG, www.ffg.at) as part of
CropShift.

<a href="https://www.ffg.at/">
<img src="https://www.ffg.at/sites/default/files/allgemeine_downloads/Logos_2018/FFG_Logo_EN_RGB_1000px.png"
alt="Logo FFG" style="width:15rem;">
</a>

## 📚 references

\[1\] Allen, R. G. (Ed.). (2000). Crop evapotranspiration: Guidelines
    for computing crop water requirements (repr). Food and
    Agriculture Organization of the United Nations.  
\[2\] Eitzinger, J., Daneu, V., Kubu, G., Thaler, S., Trnka, M.,
    Schaumberger, A., Schneider, S., & Tran, T. M. A. (2024). Grid based
    monitoring and forecasting system of cropping conditions and risks
    by agrometeorological indicators in Austria – Agricultural Risk
    Information System ARIS. Climate Services, 34, 1.
    <https://doi.org/10.1016/j.cliser.2024.100478>.  
\[3\] Schaumberger, A. (2011). Räumliche Modelle zur Vegetations- und
    Ertragsdynamik im Wirtschaftsgrünland [Dissertation, Graz University
    of Technology].
    <https://repository.tugraz.at/publications/npc97-y3058>.  
