![GitHub Tag](https://img.shields.io/github/v/tag/j-haacker/aris_lite)
![GitHub top language](https://img.shields.io/github/languages/top/j-haacker/aris_lite)
![GitHub License](https://img.shields.io/github/license/j-haacker/aris_lite)

# ARIS_lite

ARIS models plant growth based on environmental parameter.


## 🌱 state

Integral parts are being developed. At this stage you have to expect
refactoring!


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
- compute daily crop-specific stress index based on maximum surface air temperature and soil water saturation
- estimate yield expectations based on stress index


## 🚀 coming soon

- consistent observable names with ECMWF climate scenario data
- paths and observable names will be modifiable via a config file
- pipelines and tutorials will be extended


## 🔗 dependencies

- view the environment.yml
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
