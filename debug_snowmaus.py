from debug_snippets import read_ctrl_data

min_air_temp = read_ctrl_data("min_air_temp")
max_air_temp = read_ctrl_data("max_air_temp")
precipitation = read_ctrl_data("precipitation")

from snowmaus import snowfall

# turns out reference "snowcover" is not snowcover. heading points to cumulative
# number of days with snowfall