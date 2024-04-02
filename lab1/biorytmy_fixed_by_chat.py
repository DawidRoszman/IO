from datetime import datetime
import math


def get_birthdate():
    while True:
        try:
            year = int(input("What's your birth year: "))
            month = int(input("What's your birth month: "))
            day = int(input("What's your birth day: "))
            birthdate = datetime(year, month, day)
            return birthdate
        except ValueError:
            print("Invalid input. Please enter valid numbers for year, month, and day.")


def calculate_days_since_birth(birthdate):
    today = datetime.today()
    delta = today - birthdate
    return delta.days


def get_user_info():
    user_info = {}
    user_info["name"] = input("What's your name: ")
    user_info["birthdate"] = get_birthdate()
    user_info["days_since_birth"] = calculate_days_since_birth(user_info["birthdate"])
    return user_info


def welcome_user(user_info):
    print(f'Welcome {user_info["name"]}!')
    print(f'Today is your {user_info["days_since_birth"]}th day.')


def calculate_physical(days):
    return math.sin((2 * math.pi / 23) * days)


def calculate_emotional(days):
    return math.sin((2 * math.pi / 28) * days)


def calculate_intellectual(days):
    return math.sin((2 * math.pi / 33) * days)


def check_day_status(status, name):
    if status > 0.5:
        print(f"You're having a good {name} day today.")
    elif status < -0.5:
        print(f"You're having a bad {name} day today.")


def check_tomorrow_status(current, next):
    if next > current:
        print("Don't worry, tomorrow will be better.")


def main():
    user_info = get_user_info()
    welcome_user(user_info)
    statuses = [calculate_physical, calculate_emotional, calculate_intellectual]
    names = ["physical", "emotional", "intellectual"]
    for status, name in zip(statuses, names):
        current = status(user_info["days_since_birth"])
        check_day_status(current, name)
        next_day_status = status(user_info["days_since_birth"] + 1)
        check_tomorrow_status(current, next_day_status)


if __name__ == "__main__":
    main()
