# import pandas as pd
# # import ace_tools as tools; tools.display_dataframe_to_user(name="Tutoring Schedule Fall 2024", dataframe=df_schedule)

# # Define the data for each day of the week
# data = {
#     "Time": [
#         "8-9:15", "9:30-10:45", "11-12:15", "12:15-1", "1-2:15", "2:30-3:45", "4-5:15", "5:30-6:45"
#     ],
#     "Monday": [
#         "Atsia, Rupak, Kaleigh", "Atsia, Malik, Rupak, Kaleigh", "Atsia", "Rupak, Raman (12:30)", 
#         "Mackenzie, Daniel (1:30), Raman, Rupak", "Mackenzie, Daniel, Rupak", 
#         "Rubina, Raman, Rupak", "Rubina, Raman, Rupak"
#     ],
#     "Tuesday": [
#         "Daniel, Prasanna", "Prasanna", "", "", 
#         "Mackenzie, Daniel (1:30), Raman", "Mackenzie, Malik, Raman", 
#         "Rubina, Malik, Raman", "Raman, Malik"
#     ],
#     "Wednesday": [
#         "Atsia, Rupak, Kaleigh", "Atsia, Malik, Rupak, Kaleigh", "Atsia (12:00)", "", 
#         "Rupak, Daniel (1:30), Raman", "Daniel, Rubina, Rupak", 
#         "Rubina, Malik, Raman", "Raman, Malik"
#     ],
#     "Thursday": [
#         "Daniel, Prasanna", "Prasanna", "", "", 
#         "Mackenzie", "Malik, Raman", 
#         "Malik, Raman, Rubina, Kaleigh", ""
#     ]
# }

# # Create a DataFrame from the dictionary
# df_schedule = pd.DataFrame(data)

# # Save the DataFrame to an Excel file
# file_path = 'Tutoring_Schedule_Fall_2024.xlsx'
# df_schedule.to_excel(file_path, index=False)

# tools.display_dataframe_to_user(name="Tutoring Schedule Fall 2024", dataframe=df_schedule)
import pandas as pd

# Define the data for each day of the week, with names arranged vertically (using '\n' for new lines)
data = {
    "Time": [
        "8-9:15", "9:30-10:45", "11-12:15", "12:15-1", "1-2:15", "2:30-3:45", "4-5:15", "5:30-6:45"
    ],
    "Monday": [
        "Atsia\nRupak\nKaleigh", "Atsia\nMalik\nRupak\nKaleigh", "Atsia", "Rupak\nRaman (12:30)", 
        "Mackenzie\nDaniel (1:30)\nRaman\nRupak", "Mackenzie\nDaniel\nRupak", 
        "Rubina\nRaman\nRupak", "Rubina\nRaman\nRupak"
    ],
    "Tuesday": [
        "Daniel\nPrasanna", "Prasanna", "", "", 
        "Mackenzie\nDaniel (1:30)\nRaman", "Mackenzie\nMalik\nRaman", 
        "Rubina\nMalik\nRaman", "Raman\nMalik"
    ],
    "Wednesday": [
        "Atsia\nRupak\nKaleigh", "Atsia\nMalik\nRupak\nKaleigh", "Atsia (12:00)", "", 
        "Rupak\nDaniel (1:30)\nRaman", "Daniel\nRubina\nRupak", 
        "Rubina\nMalik\nRaman", "Raman\nMalik"
    ],
    "Thursday": [
        "Daniel\nPrasanna", "Prasanna", "", "", 
        "Mackenzie", "Malik\nRaman", 
        "Malik\nRaman\nRubina\nKaleigh", ""
    ]
}

# Create a DataFrame from the dictionary
df_schedule = pd.DataFrame(data)

# Save the DataFrame to an Excel file
file_path = 'Tutoring_Schedule_Fall_2024_vertical.xlsx'
df_schedule.to_excel(file_path, index=False)

import ace_tools as tools; tools.display_dataframe_to_user(name="Tutoring Schedule Fall 2024 (Vertical)", dataframe=df_schedule)
