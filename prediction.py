import streamlit as st
import pickle 
import numpy as np

def load_model():
    with open('saved_steps.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_language = data["le_language"]
le_genre = data["le_genre"]
le_country = data["le_country"]
le_title = data["le_title"]

def show_page():
    st.title("Movie Finder!")

    st.write("Lets find what you will watch! Start to select!")

    languages = ('Other', 'English', 'Italian', 'German', 'French', 'Russian',
       'Spanish', 'Swedish', 'Japanese', 'English, French',
       'English, Spanish', 'Mandarin', 'Portuguese', 'Turkish', 'Hindi',
       'Korean', 'Persian', 'Cantonese', 'Telugu', 'Malayalam', 'Tamil')

    genres = ('Other', 'Drama', 'Crime, Drama', 'Drama, War',
       'Crime, Drama, Mystery', 'Comedy', 'Horror',
       'Action, Adventure, Drama', 'Comedy, Drama', 'Drama, Romance',
       'Comedy, Drama, Romance', 'Comedy, Romance', 'Drama, Thriller',
       'Action, Adventure, Comedy', 'Action, Drama',
       'Action, Crime, Drama', 'Thriller', 'Crime, Drama, Thriller',
       'Horror, Thriller', 'Comedy, Crime', 'Action')

    countries = ('USA', 'Australia', 'Italy', 'Germany', 'Other', 'France',
       'Russia', 'Mexico', 'Sweden', 'Japan', 'UK', 'Soviet Union',
       'Spain', 'UK, USA', 'India', 'Brazil', 'Turkey', 'Poland',
       'Canada', 'South Korea', 'Hong Kong', 'Iran')

    years = (1894, 1906, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919,
       1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930,
       1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941,
       1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952,
       1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963,
       1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974,
       1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985,
       1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996,
       1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
       2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,
       2019, 2020)

    durations = (41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,
        54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,
        67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
        80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,
        93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105,
       106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
       119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
       132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
       145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
       158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
       171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
       184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196,
       197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
       210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 223,
       224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 237,
       238, 239, 240, 241, 242, 243, 244, 245, 247, 250, 252, 253, 255,
       257, 258, 260, 261, 262, 263, 264, 265, 267, 269, 271, 272, 275,
       278, 279, 281, 285, 287, 288, 290, 293, 298, 299, 300, 301, 302,
       303, 306, 311, 315, 317, 319, 321, 323, 328, 330, 335, 338, 345,
       357, 360, 366, 369, 398, 410, 418, 421, 439, 442, 450, 485, 540,
       570, 580, 729, 808)

    country = st.selectbox("Country", countries)
    language = st.selectbox("Language", languages)
    genre = st.selectbox("Genre", genres)
    duration = st.selectbox("Duration",durations)
    year = st.selectbox("Year",years)

    ok = st.button("Find my movie!")
    if ok:
        X = np.array([[year,duration,language,genre,country]])
        X[:,2] = le_language.transform(X[:,2])
        X[:,3] = le_genre.transform(X[:,3])
        X[:,4] = le_country.transform(X[:,4])

        movie_num = regressor.predict(X)
        movie_num = movie_num.astype(int)
        movie = np.array2string(le_title.inverse_transform(movie_num))
        st.subheader(f"Your movie is: {movie}")