
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pandas as pd


@dataclass
class GameGeneratorConfig:
    # Data size configuration
    N_USERS: int = 50_000
    N_GAMES: int = 5_000
    N_RATINGS: int = 500_000
    TIME_PERIOD_DAYS: int = 365

    # User generation parameters
    MIN_AGE: int = 13
    MAX_AGE: int = 80
    MEAN_AGE: int = 30
    AGE_STD: int = 10
    GENDERS: list[str] = field(default_factory=lambda: ["M", "F"])

    # Game categories
    CATEGORIES: list[str] = field(
        default_factory=lambda: [
            "RPG",
            "FPS",
            "Strategy",
            "Sports",
            "Puzzle",
            "Adventure",
            "Simulation",
            "Fighting",
            "Platform",
            "Racing",
        ]
    )

    # Difficulty levels
    DIFFICULTIES: list[str] = field(
        default_factory=lambda: ["Easy", "Medium", "Hard", "Expert"]
    )

    # Age-based category preferences - More extreme preferences
    AGE_CATEGORY_PREFS: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "young": {
                "RPG": 1.4,
                "FPS": 1.6,
                "Strategy": 0.5,
                "Sports": 1.2,
                "Puzzle": 0.4,
                "Adventure": 1.3,
                "Simulation": 0.6,
                "Fighting": 1.5,
                "Platform": 1.4,
                "Racing": 1.3,
            },
            "adult": {
                "RPG": 1.2,
                "FPS": 0.6,
                "Strategy": 1.6,
                "Sports": 0.9,
                "Puzzle": 1.5,
                "Adventure": 1.0,
                "Simulation": 1.4,
                "Fighting": 0.5,
                "Platform": 0.4,
                "Racing": 0.7,
            },
        }
    )

    # Gender-based category preferences - Stronger preferences
    GENDER_CATEGORY_PREFS: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "M": {
                "FPS": 1.5,
                "Sports": 1.4,
                "Strategy": 1.3,
                "Fighting": 1.4,
                "Racing": 1.3,
                "Puzzle": 0.7,
                "Simulation": 0.8,
            },
            "F": {
                "Puzzle": 1.4,
                "Adventure": 1.3,
                "Simulation": 1.4,
                "RPG": 1.3,
                "FPS": 0.7,
                "Fighting": 0.8,
                "Sports": 0.8,
            },
        }
    )

    # Difficulty preferences by age - More pronounced preferences
    AGE_DIFFICULTY_PREFS: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "young": {"Easy": 0.6, "Medium": 1.1, "Hard": 1.4, "Expert": 1.6},
            "adult": {"Easy": 1.3, "Medium": 1.4, "Hard": 0.8, "Expert": 0.5},
        }
    )

    # Rating parameters
    MIN_RATING: int = 1
    MAX_RATING: int = 5
    RATING_NOISE_STD: float = 0.8  # Increased noise for more variance
    BASE_RATING: float = 3.0  # New parameter for base rating

    # Age thresholds
    YOUNG_AGE_THRESHOLD: int = 25


@dataclass
class User:
    id: UUID
    age: int
    gender: str


@dataclass
class Game:
    id: UUID
    category: str
    difficulty: str
    created_at_ms: int


@dataclass
class Rating:
    user_id: UUID
    game_id: UUID
    rating: int
    timestamp_ms: int


class GameDataGenerator:
    def __init__(self, config: GameGeneratorConfig = GameGeneratorConfig()):
        self.config = config
        self.end_date: datetime = datetime.now()
        self.start_date: datetime = self.end_date - timedelta(
            days=config.TIME_PERIOD_DAYS
        )

    def generate_users(self) -> pd.DataFrame:
        users: list[User] = []
        for _ in range(self.config.N_USERS):
            age = int(np.random.normal(self.config.MEAN_AGE, self.config.AGE_STD))
            age = max(self.config.MIN_AGE, min(age, self.config.MAX_AGE))
            users.append(
                User(id=uuid4(), age=age, gender=random.choice(self.config.GENDERS))
            )
        return pd.DataFrame([vars(u) for u in users])

    def generate_games(self) -> pd.DataFrame:
        games: list[Game] = []
        for _ in range(self.config.N_GAMES):
            category = random.choice(self.config.CATEGORIES)
            difficulty = random.choice(self.config.DIFFICULTIES)

            created_at = self.start_date + timedelta(
                seconds=random.randint(
                    0, int((self.end_date - self.start_date).total_seconds())
                )
            )

            games.append(
                Game(
                    id=uuid4(),
                    category=category,
                    difficulty=difficulty,
                    created_at_ms=int(created_at.timestamp() * 1000),
                )
            )
        return pd.DataFrame([vars(g) for g in games])

    def calculate_rating_preference(
        self,
        game: dict,
        user: dict,
    ) -> int:
        # Determine age group with more granularity
        if user["age"] < 18:
            age_factor = 1.4  # Teenagers
        elif user["age"] < 25:
            age_factor = 1.2  # Young adults
        elif user["age"] < 40:
            age_factor = 1.0  # Adults
        else:
            age_factor = 0.8  # Older adults

        age_group = (
            "young" if user["age"] < self.config.YOUNG_AGE_THRESHOLD else "adult"
        )

        # Category preferences with stronger impact
        category_factor = (
            self.config.AGE_CATEGORY_PREFS[age_group][game["category"]] * 1.5
        )

        # Gender preferences with stronger impact
        gender_pref = self.config.GENDER_CATEGORY_PREFS.get(user["gender"], {}).get(
            game["category"], 1.0
        )
        gender_factor = (
            gender_pref - 1.0
        ) * 1.5 + 1.0  # Amplify the difference from 1.0

        # Difficulty preferences based on age
        difficulty_factor = self.config.AGE_DIFFICULTY_PREFS[age_group][
            game["difficulty"]
        ]

        # Calculate base rating with more weight on user characteristics
        base_rating = self.config.BASE_RATING * (
            category_factor * 0.4  # Increased category importance
            + gender_factor * 0.3  # Increased gender importance
            + age_factor * 0.2  # Age factor
            + difficulty_factor * 0.1  # Reduced difficulty importance
        )

        # Add noise that scales with the base rating
        noise = np.random.normal(0, self.config.RATING_NOISE_STD) * (base_rating / 3.0)

        # Calculate final rating
        final_rating = max(
            self.config.MIN_RATING, min(base_rating + noise, self.config.MAX_RATING)
        )

        return int(round(final_rating))

    def generate_ratings(
        self, users_df: pd.DataFrame, games_df: pd.DataFrame
    ) -> pd.DataFrame:
        ratings: list[Rating] = []

        users_dict = users_df.to_dict("records")
        games_dict = games_df.to_dict("records")

        # Create lookup dictionaries
        users_by_id = {str(u["id"]): u for u in users_dict}
        games_by_id = {str(g["id"]): g for g in games_dict}

        start_time = min(games_df["created_at_ms"]) // 1000
        end_time = max(games_df["created_at_ms"]) // 1000

        for i in range(self.config.N_RATINGS):
            if i % 10_000 == 0:
                print(f"Processing {i}/{self.config.N_RATINGS}")
            user_id = random.choice(list(users_by_id.keys()))
            game_id = random.choice(list(games_by_id.keys()))

            user = users_by_id[user_id]
            game = games_by_id[game_id]

            timestamp = random.uniform(
                max(start_time, game["created_at_ms"] / 1000), end_time
            )

            rating_value = self.calculate_rating_preference(game, user)

            ratings.append(
                Rating(
                    user_id=UUID(user_id),
                    game_id=UUID(game_id),
                    rating=rating_value,
                    timestamp_ms=int(timestamp * 1000),
                )
            )

        return pd.DataFrame([vars(r) for r in ratings])

    def generate_all_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("Generating users...")
        users_df = self.generate_users()

        print("Generating games...")
        games_df = self.generate_games()

        print("Generating ratings...")
        ratings_df = self.generate_ratings(users_df, games_df)

        # Save to CSV
        output_dir = Path(__file__).parent / "dummy_data"
        output_dir.mkdir(exist_ok=True)

        users_df.to_csv(output_dir / "users.csv", index=False)
        games_df.to_csv(output_dir / "games.csv", index=False)
        ratings_df.to_csv(output_dir / "user_ratings.csv", index=False)

        return users_df, games_df, ratings_df


def main():
    # Example usage with custom configuration
    config = GameGeneratorConfig(
        # N_USERS=10_000,
        # N_GAMES=1_000,
        # N_RATINGS=100_000,
        # TIME_PERIOD_DAYS=180,
    )

    generator = GameDataGenerator(config)
    generator.generate_all_data()


if __name__ == "__main__":
    main()
