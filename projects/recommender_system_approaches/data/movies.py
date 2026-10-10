"""Shared movie dataset used by all three recommendation system approaches.

Using the same dataset across approaches lets us compare the three
techniques (vector/Qdrant, NLP Bag-of-Words, and TF-IDF content-based)
on equal footing, as requested when analyzing the three reference
articles.

Each movie record intentionally mirrors the schema used by the IMDB
Top-250-style dataset in the Analytics Vidhya tutorial: title, director,
a short list of lead actors, a plot synopsis, and genre tags.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Movie:
    """A single movie record shared across all three recommender approaches.

    :param title: The movie's title (used as its unique identifier here).
    :param director: The director's full name.
    :param actors: Up to three lead actors' full names.
    :param plot: A short plot synopsis.
    :param genres: A list of genre tags.
    """

    title: str
    director: str
    actors: list[str]
    plot: str
    genres: list[str]


MOVIES: list[Movie] = [
    Movie(
        "The Avengers",
        "Joss Whedon",
        ["Robert Downey Jr.", "Chris Evans", "Scarlett Johansson"],
        "Earth's mightiest heroes must come together to stop Loki and his "
        "alien army from enslaving humanity using the power of the Tesseract.",
        ["Action", "Adventure", "Sci-Fi"],
    ),
    Movie(
        "Avengers: Age of Ultron",
        "Joss Whedon",
        ["Robert Downey Jr.", "Chris Evans", "Scarlett Johansson"],
        "Tony Stark creates an artificial intelligence program called Ultron "
        "to protect Earth, but it turns against humanity, forcing the "
        "Avengers to reunite and stop it.",
        ["Action", "Adventure", "Sci-Fi"],
    ),
    Movie(
        "Captain America: Civil War",
        "Anthony Russo",
        ["Chris Evans", "Robert Downey Jr.", "Scarlett Johansson"],
        "Political pressure to regulate the Avengers splits them into two "
        "opposing factions led by Captain America and Iron Man.",
        ["Action", "Adventure", "Sci-Fi"],
    ),
    Movie(
        "The Incredible Hulk",
        "Louis Leterrier",
        ["Edward Norton", "Liv Tyler", "Tim Roth"],
        "Bruce Banner, scarred by gamma radiation that transforms him into "
        "the Hulk, hides from the military while searching for a cure.",
        ["Action", "Sci-Fi"],
    ),
    Movie(
        "Iron Man 3",
        "Shane Black",
        ["Robert Downey Jr.", "Gwyneth Paltrow", "Don Cheadle"],
        "Tony Stark must defend his loved ones from a terrorist called the "
        "Mandarin after his home is destroyed and his resources taken.",
        ["Action", "Adventure", "Sci-Fi"],
    ),
    Movie(
        "Guardians of the Galaxy",
        "James Gunn",
        ["Chris Pratt", "Zoe Saldana", "Dave Bautista"],
        "A group of intergalactic criminals must band together to stop a "
        "fanatical warrior from using a powerful orb to destroy planets.",
        ["Action", "Adventure", "Comedy", "Sci-Fi"],
    ),
    Movie(
        "Interstellar",
        "Christopher Nolan",
        ["Matthew McConaughey", "Anne Hathaway", "Jessica Chastain"],
        "A team of explorers travels through a wormhole in space in an "
        "attempt to ensure humanity's survival as Earth becomes uninhabitable.",
        ["Adventure", "Drama", "Sci-Fi"],
    ),
    Movie(
        "Inception",
        "Christopher Nolan",
        ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page"],
        "A thief who steals corporate secrets through dream-sharing is given "
        "the inverse task of planting an idea into a target's subconscious.",
        ["Action", "Sci-Fi", "Thriller"],
    ),
    Movie(
        "The Dark Knight",
        "Christopher Nolan",
        ["Christian Bale", "Heath Ledger", "Aaron Eckhart"],
        "Batman must accept one of the greatest psychological and physical "
        "tests when the Joker wreaks havoc on the people of Gotham City.",
        ["Action", "Crime", "Drama"],
    ),
    Movie(
        "The Dark Knight Rises",
        "Christopher Nolan",
        ["Christian Bale", "Tom Hardy", "Anne Hathaway"],
        "Eight years after the Joker's reign of anarchy, Batman must return "
        "to defend Gotham City from the brutal mercenary Bane.",
        ["Action", "Crime", "Drama"],
    ),
    Movie(
        "The Martian",
        "Ridley Scott",
        ["Matt Damon", "Jessica Chastain", "Kristen Wiig"],
        "An astronaut becomes stranded on Mars after his crew assumes him "
        "dead, and must find a way to survive and signal Earth for rescue.",
        ["Adventure", "Drama", "Sci-Fi"],
    ),
    Movie(
        "Gravity",
        "Alfonso Cuaron",
        ["Sandra Bullock", "George Clooney", "Ed Harris"],
        "Two astronauts work together to survive after an accident leaves "
        "them stranded in space with no way to return to Earth.",
        ["Drama", "Sci-Fi", "Thriller"],
    ),
    Movie(
        "The Shawshank Redemption",
        "Frank Darabont",
        ["Tim Robbins", "Morgan Freeman", "Bob Gunton"],
        "Two imprisoned men bond over a number of years, finding solace and "
        "eventual redemption through acts of common decency.",
        ["Drama"],
    ),
    Movie(
        "The Godfather",
        "Francis Coppola",
        ["Marlon Brando", "Al Pacino", "James Caan"],
        "The aging patriarch of an organized crime dynasty transfers control "
        "of his clandestine empire to his reluctant son.",
        ["Crime", "Drama"],
    ),
    Movie(
        "The Godfather: Part II",
        "Francis Coppola",
        ["Al Pacino", "Robert Duvall", "Diane Keaton"],
        "The early life and career of Vito Corleone in 1920s New York is "
        "portrayed, while his son expands and tightens his grip on the family business.",
        ["Crime", "Drama"],
    ),
    Movie(
        "Pulp Fiction",
        "Quentin Tarantino",
        ["John Travolta", "Uma Thurman", "Samuel Jackson"],
        "The lives of two mob hitmen, a boxer, a gangster's wife, and a pair "
        "of diner bandits intertwine in four tales of violence and redemption.",
        ["Crime", "Drama"],
    ),
    Movie(
        "Kill Bill: Vol. 1",
        "Quentin Tarantino",
        ["Uma Thurman", "David Carradine", "Daryl Hannah"],
        "A former assassin wakes from a coma and sets out on a roaring "
        "rampage of revenge against the team of killers who betrayed her.",
        ["Action", "Crime", "Drama"],
    ),
    Movie(
        "Django Unchained",
        "Quentin Tarantino",
        ["Jamie Foxx", "Christoph Waltz", "Leonardo DiCaprio"],
        "A freed slave sets out with a German bounty hunter to rescue his "
        "wife from a brutal Mississippi plantation owner.",
        ["Drama", "Western"],
    ),
    Movie(
        "Forrest Gump",
        "Robert Zemeckis",
        ["Tom Hanks", "Robin Wright", "Gary Sinise"],
        "A slow-witted but kind-hearted man from Alabama witnesses and "
        "unwittingly influences several defining historical events of the 20th century.",
        ["Drama", "Romance"],
    ),
    Movie(
        "Cast Away",
        "Robert Zemeckis",
        ["Tom Hanks", "Helen Hunt", "Paul Sanchez"],
        "A FedEx executive must transform himself physically and emotionally "
        "to survive after being marooned on a deserted island.",
        ["Adventure", "Drama"],
    ),
    Movie(
        "Saving Private Ryan",
        "Steven Spielberg",
        ["Tom Hanks", "Matt Damon", "Edward Burns"],
        "Following the Normandy landings, a group of soldiers goes behind "
        "enemy lines to retrieve a paratrooper whose brothers were killed in action.",
        ["Drama", "War"],
    ),
    Movie(
        "Schindler's List",
        "Steven Spielberg",
        ["Liam Neeson", "Ralph Fiennes", "Ben Kingsley"],
        "In German-occupied Poland, an industrialist gradually becomes "
        "concerned for his Jewish workforce after witnessing their persecution by the Nazis.",
        ["Drama", "History", "War"],
    ),
    Movie(
        "Jurassic Park",
        "Steven Spielberg",
        ["Sam Neill", "Laura Dern", "Jeff Goldblum"],
        "A pragmatic paleontologist visiting a cloned dinosaur theme park is "
        "tasked with protecting a couple of kids after a power failure causes chaos.",
        ["Adventure", "Sci-Fi", "Thriller"],
    ),
    Movie(
        "Jaws",
        "Steven Spielberg",
        ["Roy Scheider", "Robert Shaw", "Richard Dreyfuss"],
        "A police chief, a marine biologist, and a grizzled shark hunter "
        "seek to destroy a man-eating great white shark terrorizing a beach town.",
        ["Adventure", "Drama", "Thriller"],
    ),
    Movie(
        "The Matrix",
        "Lana Wachowski",
        ["Keanu Reeves", "Laurence Fishburne", "Carrie-Anne Moss"],
        "A computer hacker learns from mysterious rebels about the true "
        "nature of his reality and his role in the war against its controllers.",
        ["Action", "Sci-Fi"],
    ),
    Movie(
        "The Matrix Reloaded",
        "Lana Wachowski",
        ["Keanu Reeves", "Laurence Fishburne", "Carrie-Anne Moss"],
        "Neo and his allies race against time before the machines discover "
        "the location of Zion, the last human city, and destroy it.",
        ["Action", "Sci-Fi"],
    ),
    Movie(
        "John Wick",
        "Chad Stahelski",
        ["Keanu Reeves", "Michael Nyqvist", "Alfie Allen"],
        "An ex-hitman comes out of retirement to track down the gangsters "
        "that killed his dog and took everything from him.",
        ["Action", "Crime", "Thriller"],
    ),
    Movie(
        "Speed",
        "Jan de Bont",
        ["Keanu Reeves", "Dennis Hopper", "Sandra Bullock"],
        "A young police officer must prevent a bomb exploding aboard a city "
        "bus by keeping its speed above 50 mph.",
        ["Action", "Thriller"],
    ),
    Movie(
        "Toy Story",
        "John Lasseter",
        ["Tom Hanks", "Tim Allen", "Don Rickles"],
        "A cowboy doll is profoundly threatened and jealous when a new "
        "spaceman action figure supplants him as top toy in a boy's room.",
        ["Animation", "Adventure", "Comedy"],
    ),
    Movie(
        "Finding Nemo",
        "Andrew Stanton",
        ["Albert Brooks", "Ellen DeGeneres", "Alexander Gould"],
        "After his son is captured in the Great Barrier Reef and taken to "
        "Sydney, a timid clownfish sets out on a journey to bring him home.",
        ["Animation", "Adventure", "Comedy"],
    ),
]
