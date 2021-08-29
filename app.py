import os, re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import base64
from collections import defaultdict
import zipfile
import random
import io
from PIL import Image
from collections import defaultdict
import altair as alt


from utilities import SessionState  # Assuming SessionState.py lives on this folder

# remove uploader warning
st.set_option('deprecation.showfileUploaderEncoding', False)

# get states from this session, to enable button functionality

state = SessionState._get_state()

# ----------------------Utility functions----------------------
def get_file_content_as_string(path):
    # Download a single file and make its content available as a string.
    url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """

    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    
    # some strings <-> bytes conversions necessary here
    if isinstance(object_to_download, bytes):
        b64 = base64.b64encode(object_to_download).decode()
    else:
        b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


def df_to_excel_bytes(df_dict):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    for sheet_name, df in df_dict.items():
        df.to_excel(writer, sheet_name=sheet_name)
    writer.save()
    return output.getvalue()


def zip_images(figures):
    """Saves collection of images to zip file in memory, returns zipfile object for later download.

    Parameters
    ----------
    figures : [dict]
        Dictionary mapping figure_name: figure object.
    """    
    memory_zip = io.BytesIO()
    zf = zipfile.ZipFile(memory_zip, mode="w")
    for fig_name, fig in figures.items():
        buf = io.BytesIO()
        fig.savefig(buf)
        plt.close(fig)
        img_name = f'{fig_name}.png'
        zf.writestr(img_name, buf.getvalue())
    zf.close()
    
    return zf, memory_zip
    

def zip_download(memory_zip, filename):
    # find beginning of file
    memory_zip.seek(0)
    #read the data
    data = memory_zip.read()
    tmp_download_link = download_link(data, f'{filename}.zip', 'Click here to download!')

    return tmp_download_link


def insert_new_line(num=1):
    return [
        st.text("") for i in range(num)
    ]


# ----------------------Data manipulation----------------------
def upload_dataset():

    # data_file = st.file_uploader("To begin, please upload your prepared dataset.", type=['csv'])
    uploaded_files = st.file_uploader("Upload wideform datasets", type="csv", accept_multiple_files=True)
    if len(uploaded_files) < 1:
        return False
    for file in uploaded_files:
        file.seek(0)
    data = pd.concat([pd.read_csv(file) for file in uploaded_files]).rename(columns={'molecule_number': 'sample_name'})
    data.drop([col for col in data.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

    state.data_paths = uploaded_files
    state.dataset = data
    if st.button('Validate'):
        return data    
      

def validate_data(data):

    data_preview = st.beta_expander("Preview uploaded dataset", expanded=False)
    with data_preview:       
        st.dataframe(data)

    summary = st.beta_expander("Dataset Summary", expanded=True)
    with summary:
        # preview info
        st.markdown(f"Total number of samples: {data.shape[0]}")
        st.markdown(f"Number of timepoints: {len([col for col in data.columns.tolist() if col not in ['sample_name', 'label']])}")

    
    labels = st.beta_expander("Labels", expanded=True)
    col1, col2 = st.beta_columns(2)
    with labels:
        if 'label' in data.columns.tolist():
            # preview task info
            unique_labels = data['label'].dropna().unique().tolist()
            st.markdown(f"Label column detected. Current labels: {unique_labels}")
            st.markdown(f"{len(data['label']) - len(data['label'].dropna())} samples still to be labelled.")
            state.labels = unique_labels
            state.num_labels = len(unique_labels)
        else:
            state.labels = None
            state.num_labels = None
            st.markdown(f"No label column detected. If you expect a label column to be detected, ensure the relevant column header is 'label'. Otherwise, labels will be specified in the next step.")

    download_options = st.beta_expander("Download summary", expanded=False)
    with download_options:
        col1, col2 = st.beta_columns(2)
        if col1.button("Download compiled dataset"):
            tmp_download_link = download_link(df_to_excel_bytes(data), 'compiled_data.xlsx', 'Click here to download!')
            col1.markdown(tmp_download_link, unsafe_allow_html=True)

    return data


def prepare_dataset(data):

    prepared_data = pd.melt(
        data.copy(),
        id_vars='sample_name', 
        value_vars=[col for col in data.columns.tolist() if col not in ['sample_name', 'label']], 
        var_name='time', 
        value_name='value'
        )
    prepared_data['time'] = prepared_data['time'].astype(int)

    state.data_prepared = prepared_data

    return prepared_data


def create_labels():

    labels = state.labels
    state.num_labels = st.text_input('How many label categories would you like to create?', value=state.num_labels)

    if state.num_labels != 'None':
        num_labels = int(state.num_labels)
        # st.write('Number of labels:', num_labels)
        st.markdown('Enter a simple name for each class. Usually this will be numbers, or a simple descriptive word.')
        labels = [
            st.text_input(
                f'Label {label+1}',
                value=f'{label+1}',
                max_chars=20,
                )
            for label in range(num_labels)
        ]
        if len(set(labels)) != len(labels):
            st.markdown('Non-unique labels identified. Please ensure all labels are unique!')
            return
        col1, col2 = st.beta_columns(2)
        if col2.button('Submit labels'):
            state.labels_chosen = True
            state.labels = labels


def read_existing_labels():
    class_labels = state.dataset[['sample_name', 'label']].copy().dropna()
    class_labels['value'] = [True for val in range(len(class_labels))]
    class_labels = pd.pivot(class_labels, index='sample_name', columns='label', values='value').fillna(False).reset_index()

    state.class_labels = {sample_name: class_labels.tolist() for sample_name, class_labels in class_labels.set_index('sample_name').iterrows()}

def prepare_labels():

    if state.labels_chosen:
        return
    if state.labels is not None:
        labels = state.labels
        st.markdown(f'You currently have the following labels: {state.labels}.')
        if len(set(labels)) != len(labels) or len(labels) < 2:
            st.markdown('Unfortunately these labels are not compatible with the classification process. Please ensure that you have more than one unique label!')
            st.markdown("To discard these labels, click 'Create new labels'.")
        else:
            st.markdown("These labels are compatible with the classification process. If you would like to create new labels anyway, you can do so using the 'Create new labels' button. Otherwise, you may submit your labels.")

        col1, col2 = st.beta_columns(2)
        if col1.button('Create new labels'):
            state.labels = None
            state.num_labels = None
            data = state.dataset
            data['label'] = np.nan
            state.dataset = data
            create_labels()
        if col2.button('Submit labels'):
            read_existing_labels()
            state.labels_chosen = True
            state.labels = labels
            return labels
    else: 
        create_labels()
  

def convert_labels():
    class_labels = pd.DataFrame(state.class_labels).T
    class_labels.columns = state.labels
    class_labels.index.name = 'sample_name'
    class_labels.reset_index(inplace=True)
    class_labels['label'] = [state.labels[x] for x in np.argmax(class_labels[state.labels].values, axis=1)]
    if 'labels' in state.dataset.columns.tolist():
        state.dataset = state.dataset.drop('label', axis=1)
    data = pd.merge(state.dataset, class_labels[['sample_name', 'label']], on=['sample_name'], how='left')

    state.data_labelled = pd.melt(
        data,
        id_vars=['sample_name', 'label'],
        value_vars=[col for col in data.columns.tolist() if col not in ['sample_name', 'label']],
        var_name='time',
        value_name='value')

    return data


def label_series(data):

    invalid_labels = []
    for sample_name, df in data.groupby('sample_name'):
        st.empty()
        col1, col2 = st.beta_columns(2)

        # # ----Plot individual timeserie-----

        fig = plot_individuals(df, sample_name=sample_name)
        col1.pyplot(fig)

        col2.markdown('Please select a label:')
        if sample_name in state.class_labels.keys():
            existing_labels = state.class_labels[sample_name]
        else:
            existing_labels = [False for option in state.labels]
        
        selected_labels = [col2.checkbox(f'{option}', key=f'{sample_name}_{option}', value=existing_labels[x]) for x, option in enumerate(state.labels)]

        if sum(selected_labels) != 1: # allow only one label!
            invalid_labels.append(sample_name)
            st.markdown(f'No valid label selection detected for {sample_name}.')
        else:
            state.class_labels[sample_name] = selected_labels

    col1, col2, col3 = st.beta_columns(3)
    if state.current_chunk > 0:
        if col1.button('Previous'):
            state.current_chunk += -1
    if len(invalid_labels) == 0:
        if col3.button('Next'):
            state.current_chunk += 1
    if col2.button('Save progress'):
        state.partial_data = convert_labels()
        tmp_download_link = download_link(state.partial_data, 'partial_data.csv', 'Click here to download!')
        col2.markdown(tmp_download_link, unsafe_allow_html=True)



# -----------------------Define plot elements-----------------------

def plot_individuals(df, sample_name, label_col=False):

    fig, ax = plt.subplots()
    sns.lineplot(data=df, x='time', y='value', color='black')
    plt.ylabel('Value')
    plt.xlabel('Time')
    plt.title(sample_name)
    # plt.savefig(f'{output_folder}{name}_v_class.png', bbox_inches='tight')

    return fig


def plot_label_summary(label_col='label'):

    if len(state.labels) < 8:
        palette = dict(zip(state.labels, ['firebrick', 'darkorange', 'rebeccapurple', 'forestgreen', 'steelblue', 'mediumvioletred', 'black', 'grey']))
    else:
        palette = sns.color_palette("flare", n_colors=len(state.labels))

    df = state.data_labelled
    df['time'] = df['time'].astype(int)
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x='time', y='value', hue=label_col, palette=palette, ci='sd')
    # replace legend labels
    plt.legend()
    plt.ylabel('Value')
    ax.set_xlabel('Time')

    return fig
        


def plot_interactive_labels(sample_names, label_col='label', container=False):

    if len(state.labels) < 8:
        palette = dict(zip(state.labels, ['firebrick', 'darkorange', 'rebeccapurple', 'forestgreen', 'steelblue', 'mediumvioletred', 'black', 'grey']))
    else:
        palette = sns.color_palette("flare", n_colors=len(state.labels))

    df = state.data_labelled[state.data_labelled['sample_name'].isin(sample_names)].copy()
    df['time'] = df['time'].astype(int)

    # fig, ax = plt.subplots()
    # for sample_name, data in df.groupby('sample_name'): 
    #     sns.lineplot(data=data, x='time', y='value', hue=label_col, palette=palette, ci='sd')
    # # replace legend labels
    # plt.legend()
    # plt.ylabel('Value')
    # ax.set_xlabel('Time')

    if container:
        container.altair_chart(alt.Chart(df).mark_line().encode(
            x='time:Q', y='value:Q', color=alt.Color('label:N', scale=alt.Scale(domain=list(palette.keys()), range=list(palette.values()))), tooltip=['sample_name'], detail='sample_name:N', ).interactive(), use_container_width=True)
    else:
        st.altair_chart(alt.Chart(df).mark_line().encode(
            x='time:Q', y='value:Q', color=alt.Color('label:N', scale=alt.Scale(domain=list(palette.keys()), range=list(palette.values()))), tooltip=['sample_name'], detail='sample_name:N', ).interactive(), use_container_width=True)


# -----------------------Define main pages-----------------------

def run_introduction():
    banner = Image.open('utilities/banner.png')
    st.image(banner)
    st.markdown("<meta charset='UTF-8'><i>A &#9889; powerfully simply method for manual classification of time series training datasets.</i>", unsafe_allow_html=True)

    example_data = pd.read_csv('utilities/example_data.csv')
    
    example_link = download_link(example_data, 'utilities/example_data.csv', 'example_data')

    
    st.markdown(
    f"""

    ## Instructions

    """)
    getting_started = st.beta_expander("Getting started", expanded=False)
    with getting_started:
        st.markdown(
        f"""
        To get started, you can explore the app functionality on a fictitious dataset using the {example_link}. To process your own data, it is first necessary to compile your time series of interest and preformat it as long-form csv files. Detailed instructions on preparing the dataset for upload are included below, and should be followed to ensure proper processing of your dataset.

        The dataset should contain only the following columns:
        1) *sample_name*; containing unique identifier for each time series,
        2) *label*; *optional*, to include some pre-labelled samples a label column containing short unique category identifiers may be included. If included, you will be given the option to keep samples pre-assigned to an identifier.
        3+) 0, 1, 2, 3 ... ; Time columns, these remaining columns should contain individual timepoint data for each sample. Columns should only be specified with integer increments (no string labels.)

        
        """, unsafe_allow_html=True)

    data_instructions = st.beta_expander("Prepare data", expanded=False)
    with data_instructions:
        st.markdown(
        f"""
        After preparing your dataset, proceed to the "Process data" functionality using the left navigational pane. This function will guide you through processing the prepared dataset. Once satisfied with the uploaded data, click continue to finalise the data validation process. Optionally, you may download a clean and processed version of the dataset which can be reuploaded to repeat any subsequent analysis steps.
        """, unsafe_allow_html=True)

    label_instructions = st.beta_expander("Prepare labels", expanded=False)
    with label_instructions:
        st.markdown(
        f"""
        After preparing the dataset, you may then proceed to specify your labels of interest. If the optional 'label' column was included, you will be presented with the option to continue with the previous label sets, or create a new label set. If the optional column was not included, you will proceed directly to the label creation phase.

        Remember to include at least 2 (or more) unique labels. Ideally, numeric or short text labels work best.
        """, unsafe_allow_html=True)

    classify_instructions = st.beta_expander("Classify data", expanded=False)
    with classify_instructions:
        st.markdown(
        f"""
        Once the data and labels have been specified, you will be able to indicate the appropriate label for each dataset. This will be presented in chunks to enable easier processing of large datasets, and once each dataset has an allocated label the 'Next' button will appear to display the following set of samples. At any time you can navigate back to previously annotated samples using the 'Previous' button. Additionally, in the case where you are processing a large dataset it is possible to generate a partially-processed summary for download. In this way, you can reload the partially processed dataset to continue labelling the remaining samples if at any point you need to delay/restart the labelling process.

        Once all datasets have been annotated, then a summary of the average value for each of the labels is shown. From here you have the option to revisit any of the samples using the 'Previous' button. 
        """, unsafe_allow_html=True)
    
    download_instructions = st.beta_expander("Summary and data download", expanded=False)
    with download_instructions:
        st.markdown(
        f"""
        Finally, you may proceed to the summary page, where you will find some brief dataset statistics and the option to display a plot for individual datasets coloured according to label type. From here, you can then download the final annotated dataset for use in later analyses. The download format mirrors that of the upload, containing a *sample_name* column, a *label* column and the remaining timepoint columns.
        """, unsafe_allow_html=True)

    troubleshooting = st.beta_expander("Troubleshooting", expanded=False)
    with troubleshooting:
        st.markdown(
        f"""
        Most errors are the result of inconsistencies in the data format provided compared to that required. If you receive an error message during the upload or labelling processes, please carefully check the dataset and labels you have provided to ensure it matches the template directions. You can also test the overall app functionality using the {example_link}.

        """, unsafe_allow_html=True)

    insert_new_line(num=5)

    st.markdown(
    f"""
    <small>*Disclaimer: the data collected and generated by this platform are not stored or retained, however no guarantee is provided on the end-to-end security of any uploaded or downloaded information. In addition, the information generated here is provided on an “as is” basis with no guarantees of completeness, accuracy, or usefulness. Any action you take as a result of this information is done so at your own risk.*</small>
        
            """, unsafe_allow_html=True
        )
    

def run_labelling():
                    
    if state.__getattr__('class_labels') == None:
        state.class_labels = {}
    if state.__getattr__('valid_labels') == None:
        state.valid_labels = True
    if state.__getattr__('current_chunk') == None:
        state.current_chunk = 0

    header = st.empty()
    instructions = st.empty()
    selection = st.empty()
    plot = st.empty()
    col1, col2, col3 = st.beta_columns(3)

    n = 10
    chunked_samples = [state.data_prepared['sample_name'].unique().tolist()[i:i+n] for i in range(0,len(state.data_prepared['sample_name'].unique().tolist()),n)]
    if state.current_chunk < len(chunked_samples):
        instructions.markdown("Select the appropriate label using the check boxes in the right-hand column. You may only select one label per series.")
        df = state.data_prepared[state.data_prepared['sample_name'].isin(chunked_samples[state.current_chunk])]
        label_series(df)
    else:
        if state.submit_labels is not None:
            st.subheader(f"Congratulations! Labels collected for {len(state.class_labels)} series. Proceed to the 'Summary' tab to view your results and download the labelled dataset.")
        else:
            header.subheader("You have now assigned labels to all time series.")
            instructions.markdown("You may review assigned labels below, and adjust labels using the 'Previous' button to go back to the appropriate position. Once satisfied, submit your labels to generate a downloadable version.")
            if state.data_labelled is None:
                # if it hasn't already been done, convert the true/False lists to labels
                convert_labels()
            state.sample_names = selection.multiselect('Please select datasets to visualise their label status', options=state.data_labelled['sample_name'].unique().tolist(),)
            if state.sample_names != [None]:
                plot_interactive_labels(state.sample_names, label_col='label', container=plot)

            if col1.button('Previous'):
                    state.current_chunk += -1
            if col3.button('Submit labels'):
                state.submit_labels = True
                convert_labels()
                

    state.sync()


def run_summary():

    st.subheader("Labelling summary")
    fig = plot_label_summary()
    st.pyplot(fig)

    summary_stats = st.beta_expander("Summary stats", expanded=False)
    with summary_stats:
        st.markdown(f"Total number of annotated datasets: {len(state.data_labelled['sample_name'].unique())}")
        st.markdown("This included:")
        col1, col2 = st.beta_columns([1, 5])
        for label, df in state.data_labelled.groupby('label'):
            col2.markdown(f"{len(df['sample_name'].unique())} samples labelled with {label}")

    interactive_visuals = st.beta_expander("Visualise individual labels", expanded=False)
    with interactive_visuals:
        state.sample_names = st.multiselect('Please select datasets to visualise their label status', options=state.data_labelled['sample_name'].unique().tolist())
        if state.sample_names != [None]:
            plot_interactive_labels(state.sample_names, label_col='label')


    download_options = st.beta_expander("Download", expanded=False)
    with download_options:
        data = pd.pivot(state.data_labelled, index=['sample_name', 'label'], columns=['time'], values=['value'])
        data.columns = data.columns.droplevel(0)
        data.reset_index(inplace=True)
        st.dataframe(data)
        col1, col2 = st.beta_columns(2)
        if col1.button("Download compiled dataset"):
            tmp_download_link = download_link(data, 'labelled_data.csv', 'Click here to download!')
            col1.markdown(tmp_download_link, unsafe_allow_html=True)


# -----------------------Main run function-----------------------
def main():

    st.markdown(
            f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: 2000px;
            padding-top: 1rem;
            padding-right: 10rem;
            padding-left: 10rem;
            padding-bottom: 10rem;
        }}
    </style>
    """,
            unsafe_allow_html=True,
        )



    app_mode = st.sidebar.selectbox("Choose functionality:",
        ["Home", "Prepare dataset", "Prepare labels", "Classify training data", "Summary"]) #, "Predict labels for new data", "Validate predicted labels", 'Help & FAQ'])

    if not hasattr(state, 'data_processed'):
        state.data_processed = False
    if not hasattr(state, 'labels_chosen'):
        state.labels_chosen = False
    if not hasattr(state, 'labels'):
        state.labels = False
    if not hasattr(state, 'num_labels'):
        state.num_labels = False
    if not hasattr(state, 'new_labels'):
        state.new_labels = False

    icon = Image.open('utilities/icon.png')

    if app_mode == 'Home':
        run_introduction()


    if app_mode == 'Prepare dataset':
        col1, col2 = st.beta_columns([1, 5])
        col1.image(icon)
        col2.markdown("# Time Series Labelling App")
        if type(state.dataset) == pd.DataFrame:
            data = state.dataset
            data = validate_data(data)
            if st.button('Continue'):
                prepared_data = prepare_dataset(data)
                state.data_processed = True
        else:
            data = upload_dataset()
        if state.data_processed == True:
            st.subheader("Congratulations! Your data has been compiled. You may now proceed to prepare your labels using the navigation bar to the left.")


    if app_mode == 'Prepare labels':
        col1, col2 = st.beta_columns([1, 5])
        col1.image(icon)
        col2.markdown("# Time Series Labelling App")
        if state.data_processed and not state.labels_chosen:
            labels = prepare_labels()
        elif state.labels_chosen:
            st.subheader(f"Congratulations! You have created the following labels: {state.labels}")
            st.subheader("You may now proceed to the classification tab.")
        else:
            st.subheader("*Oh no! Did you forget a step?*")
            st.markdown("To begin, you must validate your prepared dataset. \n Please select 'Prepare data'.")


    elif app_mode == "Classify training data":
        col1, col2 = st.beta_columns([1, 5])
        col1.image(icon)
        col2.markdown("# Time Series Labelling App")
        if state.data_processed is not None and state.labels_chosen is not None:
            labelling = run_labelling()

        elif state.data_processed is None:
            st.subheader("*Oh no! Did you forget a step?*")
            st.markdown("During validation, you must select your labels for classification. \n Please return to the 'Prepare labels' pane.")

        else:
            st.subheader("*Oh no! Did you forget a step?*")
            st.markdown("To begin, you must validate your prepared dataset. \n Please return to the 'Prepare dataset' pane.")

    elif app_mode == "Summary":
        col1, col2 = st.beta_columns([1, 5])
        col1.image(icon)
        col2.markdown("# Time Series Labelling App")
        if state.submit_labels is not None:
            run_summary()
        else:
            st.subheader("*Oh no! Did you forget a step?*")
            st.markdown("To view a summary, you must complete the classification step. \n Please return to the 'Classify training data' pane.")

    state.sync()







if __name__ == "__main__":
    main()
