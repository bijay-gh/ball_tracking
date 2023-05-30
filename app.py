from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

app = Dash(__name__)

# df1 = pd.read_csv('')
# df2 = pd.read_csv('')
# df3 = pd.read_csv('')
input_vid = 'https://github.com/mgunjan67/mgunjan67.github.io/blob/main/assets/img/LBW.gif?raw=true'
output_vid = 'https://github.com/mgunjan67/mgunjan67.github.io/blob/main/assets/img/LBW.gif?raw=true'
output_img = 'https://github.com/mgunjan67/mgunjan67.github.io/blob/main/assets/img/LBW.gif?raw=true'

app.layout = html.Div([
    html.H1(children='Ball Tracking and DRS', style={'textAlign':'center'}),
    html.H2("By Bijaya Ghimire, Diwas Lamichhane and Gunjan Kumar Mishra", style={'textAlign':'center'}),
    html.H2("Input Video"),
    html.Img(src=input_vid),
    html.H2("Output Video"),
    html.Img(src=output_vid),
    html.H2("Output Image"),
    html.Img(src=output_img),
    html.H2("Graph1"),
    # dcc.Graph(figure=px.line(df1[""])),
    html.H2("Graph2"),
    # dcc.Graph(figure=px.line(df2[""])),
    html.H2("Graph3"),
    # dcc.Graph(figure=px.line(df3[""])),
])

if __name__ == '__main__':
    app.run_server(debug=True)