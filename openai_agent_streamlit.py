from langchain.agents import create_csv_agent
from langchain.llms import OpenAI 
from dotenv import load_dotenv
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

def chat_with_gpt3(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,  # Adjust the response length as needed
    )
    return response.choices[0].text.strip()

def chat_with_csv_agent(prompt,file_name):
    load_dotenv()
    csv_agent = create_csv_agent(OpenAI(temperature=0),file_name, verbose=True)
    return csv_agent.run(prompt)

def generate_response(prompt,uploaded_file):
    load_dotenv()
    csv_agent = create_csv_agent(OpenAI(temperature=0),uploaded_file.name, verbose=True)
    response = csv_agent.run(prompt)
    st.info(response)
    dataset = pd.read_csv(uploaded_file,parse_dates=['Date'],date_format='MM/DD/YYYY')
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

def recommend_graph(data):
    # Simple recommendation based on data characteristics
    # You can customize this function based on specific requirements
    num_columns = len(data.columns)
    if num_columns == 1:
        return 'histogram'
    elif num_columns == 3:
        return 'scatter plot line plot bar plot'
    else:
        return 'bar plot'

def generate_line_plot(data, x_data,y_data,plot_name = "Line Plot"):
    date_column = find_date_column(data)
    if False:
        for column in y_data.columns:
            #data.plot(x=date_column, y=data.columns.difference([date_column]))
            plt.plot(x_data,y_data[column],label=column)
        plt.xlabel(date_column)
        plt.ylabel(data.columns.difference([date_column])[0])
    else:
        data.plot(kind='line',x='Date')
        plt.xlabel(data.columns[0])
        plt.ylabel(data.columns[1])
    plt.title(plot_name)
    plt.show()
        

def generate_scatter_plot(data, x_data, y_data, plot_name = "Scatter Plot"):
    date_column = find_date_column(data)
    color_list = ['purple','orange','blue','red','green','black']
    colornum=1
    if True:
        for column in y_data.columns:
        #data.plot(x=date_column, y=data.columns.difference([date_column]), kind='scatter')
            plt.scatter(x_data,y_data[column],c=color_list[colornum])
            colornum=colornum+1
        plt.xlabel(date_column)
        plt.ylabel(data.columns.difference([date_column])[0])
    else:
        data.plot(kind='scatter', x='Date')
        plt.xlabel(data.columns[0])
        plt.ylabel(data.columns[1])
    colornum=0
    #plt.legend()
    plt.title(plot_name)
    plt.show()

def generate_bar_plot(data):
    data.plot(kind='bar',x='Date')
    plt.xlabel("X-axis Label")
    plt.ylabel("Y-axis Label")
    plt.title("Bar Plot")
    plt.show()

def generate_pie_chart(data):
    # Assuming data contains only one column
    data.plot(kind='pie', y='column_name')
    plt.ylabel("")
    plt.title("Pie Chart")
    plt.show()

def get_chart_name():
    chart_name_prompt = input("Do you want to name this chart?")
    if chart_name_prompt in ['y','yes']:
        plot_name = input("Enter a name for the chart:")
        plt.title(plot_name)
        plt.show()
    else:
        pass

def main():
    #load_dotenv()
    #print("Chat with Chatbot. Type 'exit' to end the conversation.")
    #user_input = input("You: ")
    #chat_history = ["Welcome, how can I assist you today?"]
    #user_input = st.text_input("You: ",value="")
    #submit_button = st.button("Submit")
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
                    generate_response(text,uploaded_file)
                else:
                    generate_response_ai(text)
    #     if 'load data' in user_input.lower():
    #         file_name = input("Enter the CSV file name:")
    #         print(f"Loading data from {file_name}...")
    #         data = load_data_from_csv(file_name)
    #         x_data= data.iloc[:,0]
    #         y_data = data.iloc[:,1:]
    #     elif 'show data' in user_input.lower():
    #         print("Chatbot: Sure here's the data:")
    #         print(data)
    #     elif 'recommend graph' in user_input.lower():
    #         # Recommend graph type based on data
    #         graph_type = recommend_graph(data)
    #         if graph_type == 'histogram':
    #             print("You may consider using a line plot or histogram.")
    #         elif graph_type == 'scatter plot line plot':
    #             print("You may consider using a scatter plot or line plot.")
    #         elif graph_type == 'bar plot pie chart':
    #             print("You have multiple columns, consider using bar plots or pie charts.")
    #     elif 'create a line chart' in user_input.lower():
    #         generate_line_plot(data,x_data,y_data)
    #     elif 'create a scatter chart' in user_input.lower():
    #         generate_scatter_plot(data,x_data,y_data)
    #     elif 'create a bar chart' in user_input.lower():
    #         generate_bar_plot(data)
    #     elif 'create a pie chart' in user_input.lower():
    #         generate_pie_chart(data)
    #     elif 'automatically create a chart' in user_input.lower():
    #         plot_type = recommend_graph(data)
    #         ## print(f"Generating {plot_type}...")

    #         if 'line plot' in plot_type:
    #             generate_line_plot(data,x_data,y_data)
    #         if 'scatter plot' in plot_type:
    #             generate_scatter_plot(data,x_data,y_data)
    #         if 'bar plot' in plot_type:
    #             generate_bar_plot(data)
    #         if 'pie chart' in plot_type:
    #             generate_pie_chart(data)
    #     elif user_input.lower() in ['exit', 'quit', 'bye','end']:
    #         print("Chatbot: Goodbye!")
    #         #st.write("Goodbye!")
    #         break
    #     elif 'question about the dataset:' in user_input.lower():
    #         #df = pd.read_csv(file_name)
    #         #text_series_list = [df[col].astype(str) for col in df.columns]
    #         #text_strings = [' '.join(text_series) for text_series in text_series_list]
    #         #text_strings_string = ' '.join(map(str,text_strings))
    #         #with open(file_name) as f:
    #         #    text_strings_string = f.read() + '\n'
    #         #print(text_strings_string)
    #         prompt = f"\n You: {user_input}"
    #         response = chat_with_csv_agent(prompt,file_name)
    #         print("Chatbot:", response)
    #     else:
    #         prompt = f"You: {user_input}"
    #         response = chat_with_gpt3(prompt)
    #         print("Chatbot:", response)
    #         #chat_history.append(f"You: {user_input}")

    #         # Get the AI's reply
    #         #prompt = "\n".join(chat_history)
    #         #response = chat_with_gpt3(prompt)

    #         # Add the AI's reply to the chat history
    #         #chat_history.append("Chatbot: " + response)
    #     user_input = input("You: ")
    #     #st.text_area("Chat History","\n".join(chat_history))

    # print("Conversation ended.")

if __name__ == "__main__":
    main()
