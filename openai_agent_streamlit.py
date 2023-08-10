from langchain.agents import create_csv_agent
from langchain.llms import OpenAI 
#from dotenv import load_dotenv
import openai
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import plotly.figure_factory as ff
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.svm import SVR

#def chat_with_gpt3(prompt):
#    response = openai.Completion.create(
#        engine="text-davinci-002",
#        prompt=prompt,
#        max_tokens=150,  # Adjust the response length as needed
#    )
#    return response.choices[0].text.strip()

#def chat_with_csv_agent(prompt,file_name):
    #load_dotenv()
#    csv_agent = create_csv_agent(OpenAI(temperature=0),file_name, verbose=True)
#    return csv_agent.run(prompt)

def generate_response(prompt,uploaded_file,api_key):
    #load_dotenv()
    openai.api_key = api_key
    openai_api_key = api_key
    csv_agent = create_csv_agent(OpenAI(temperature=0,openai_api_key=api_key),uploaded_file.name, verbose=True)
    response = csv_agent.run(prompt)
    st.info(response)
    #dataset = pd.read_csv(uploaded_file,parse_dates=['Date'],date_format='MM/DD/YYYY')
    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # To read file as string:
    dataset = stringio.read()
    if 'line' in response:
        st.line_chart(data=dataset,x='Date')
    if 'bar' in response:
        st.bar_chart(data=dataset,x='Date')
    if 'distplot' in response:
        fig = generate_distplot()
        st.plotly_chart(fig[0])
        dataset = fig[1]
    if '3D regression surface' in response:
        fig = generate_regsurf()
        st.plotly_chart(fig[0])
        dataset = fig[1]
    st.dataframe(dataset)

def generate_response_ai(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,  # Adjust the response length as needed
    )
    st.info(response.choices[0].text.strip())

def generate_distplot():
    # Add histogram data
    x1 = np.random.randn(200)-2
    x2 = np.random.randn(200)
    x3 = np.random.randn(200)+2
    x4 = np.random.randn(200)+4

    # Group data together
    hist_data = [x1, x2, x3, x4]

    group_labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4']

    # Create distplot with custom bin_size
    fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .25, .5, 1])
    df = pd.DataFrame(list(zip(x1,x2,x3,x4)),columns=group_labels)
    return (fig,df)

def generate_regsurf():
    mesh_size = .02
    margin = 0

    df = px.data.iris()

    X = df[['sepal_width', 'sepal_length']]
    y = df['petal_width']

    # Condition the model on sepal width and length, predict the petal width
    model = SVR(C=1.)
    model.fit(X, y)

    # Create a mesh grid on which we will run our model
    x_min, x_max = X.sepal_width.min() - margin, X.sepal_width.max() + margin
    y_min, y_max = X.sepal_length.min() - margin, X.sepal_length.max() + margin
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)

    # Run model
    pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
    pred = pred.reshape(xx.shape)

    # Generate the plot
    fig = px.scatter_3d(df, x='sepal_width', y='sepal_length', z='petal_width')
    fig.update_traces(marker=dict(size=5))
    fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))

    return (fig,df)

def load_data_from_csv(csv_file):
    return pd.read_csv(csv_file,parse_dates=['Date'],date_format='MM/DD/YYYY')

def find_date_column(data):
    # Identify the column with datetime values, if present
    for column in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[column]):
            return column
    return None



def main():
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key",key="chatbot_api_key",type="password")
    openai.api_key = openai_api_key
    st.title("Data Whisperer's Guide to a Dashboard Revolution")
    image = Image.open('revolution.jpg')
    st.image(image,width=500)
    uploaded_file = st.file_uploader("Choose a file")
    with st.form("my_form"):
        text = st.text_area("Enter text here:")
        submitted = st.form_submit_button("Submit")
        if submitted:
            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")
                st.stop()
            else:
                if uploaded_file:
                    generate_response(text,uploaded_file,openai_api_key)
                else:
                    generate_response_ai(text)

if __name__ == "__main__":
    main()
