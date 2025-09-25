#%% packages
import streamlit as st
from helper_funs import get_movies_from_plot
# %%
st.write("Beschreibe die Handlung")

# Make the action button blue and visually integrated with the input
st.markdown(
    """
    <style>https://nodejs.org/en/download
    .stButton>button {
        background-color: #1E90FF !important; /* DodgerBlue */
        color: white !important;
        border: 1px solid #1E90FF !important;
        padding: 0.45rem 0.7rem !important;
        border-radius: 6px !important;
    }
    .stButton>button:hover {
        background-color: #1A7ED8 !important;
        border-color: #1A7ED8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Input + Button layout (per wireframe)
left_col, right_col = st.columns([10, 1], gap="small")
with left_col:
    plot_text = st.text_input(
        label="Plot",
        value="",
        placeholder="eingabefeld",
        label_visibility="collapsed",
        help=None,
        key="plot_input",
        max_chars=None,
        type="default",
        disabled=False,
        autocomplete=None,
        on_change=None,
        args=None,
        kwargs=None,
    )
with right_col:
    do_search = st.button("▶")

# Validation and search
if do_search:
    user_input = (plot_text or "").strip()
    if len(user_input) < 3:
        st.info("Bitte gib eine aussagekräftige Handlungsbeschreibung ein (mind. 3 Zeichen).")
    else:
        with st.spinner("Suche Filme..."):
            data = get_movies_from_plot(user_input)

        # Error handling
        if data.get("error"):
            st.error(f"Fehler bei der Abfrage: {data['error']}")
        else:
            movies = data.get("movies", [])
            if not movies:
                st.warning("Keine Filme gefunden.")
            else:
                for index, movie in enumerate(movies):
                    st.markdown(f"**{movie.get('title', '')}**")
                    st.write(f"Darsteller: {movie.get('main_characters', '')}")
                    st.write(f"Regisseur: {movie.get('director', '')}")
                    st.write(f"Veröffentlichungsjahr: {movie.get('release_year', '')}")
                    if index < len(movies) - 1:
                        st.divider()