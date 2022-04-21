from pages import pages

import streamlit as st
from streamlit_multipage import MultiPage

def footer(st):
    st.write("Developed by [ELC](https://elc.github.io)")


def header(st):
    st.title("Multiple Operators Neural Network")


app = MultiPage()
app.st = st


#app.start_button = "Go to the main page"
app.navbar_name = "Pages:"
app.next_page_button = "Next Chapter"
app.previous_page_button = "Previous Chapter"
app.reset_button = "Delete Cache"
app.navbar_style = "VerticalButton"

#app.navbar_style = "SelectBox"
#app.navbar_style = "HorizontalButton"


app.header = header
#app.footer = footer
#app.navbar_extra = sidebar

#app.hide_menu = True
#app.hide_navigation = True
#app.add_app("Landing", landing_page, initial_page=True)

for app_name, app_function in pages.items():
    app.add_app(app_name, app_function)

app.run()
