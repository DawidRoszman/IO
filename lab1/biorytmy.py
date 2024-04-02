# a) Napisz program w Pythonie, który spyta użytkownika o imię, rok urodzenia, miesiąc urodzenia i dzień
# urodzenia. Następnie wita użytkownika, informuje go, który dzień ich życia jest dzisiaj, oblicza ich fizyczne,
# emocjonalne i intelektualne wyniki biorytmów i drukuje je. Jeśli to możliwe, zrób to bez użycia narzędzi AI.
# Możesz korzystać z zasobów online. Sprawdź swój biorytm! Jak się czujesz dzisiaj?
# b) Zmodyfikuj swój program. Po obliczeniu biorytmów program powinien sprawdzić, czy są one wysokie (powyżej
# 0,5) i pogratulować dobrego wyniku, niskie (mniejsze niż -0,5) i pocieszyć w złym dniu. W przypadku niskiego
# wyniku algorytm sprawdza również, czy następny dzień przyniesie wyższy czy niższy wynik. W przypadku
# wyższego wyniku program powinien powiedzieć coś w rodzaju „Nie martw się. Jutro będzie lepiej!”
# c) Zmierz (mniej więcej) ile czasu spędziłeś/aś na pisaniu programu (a-b), łącznie z badaniami online. Napisz
# swoją odpowiedź jako komentarz w programie.
# d) Poproś ChatGPT lub inne narzędzie AI o korektę programu, lub poprawę stylistyczną. Zapisz kopię programu.
# e) Zacznij nową, czystą rozmowę z ChatGPT1
# (lub innym narzędziem) i spróbuj poprosić o kompletny program.
# Zobacz, czy ChatGPT będzie w stanie napisać program samodzielnie. Musisz skonstruować swoje polecenie w
# dobry, jasny, precyzyjny sposób, aby zrozumiał zadanie. Zmierz, ile czasu spędziłeś/aś na uzyskaniu
# poprawnego programu (czy zajęło to więcej czasu niż w punkcie (c)?)
# Pisanie programu zajęło mi około 40 minut.
from datetime import datetime
import math


def get_user_info():
    user_info = {}

    user_info["name"] = input("What's your name: ")
    user_info["year_of_birth"] = int(input("What's the year of your birth: "))
    user_info["month_of_birth"] = int(input("What's the month of your birth: "))
    user_info["day_of_birth"] = int(input("What's the day of your birth: "))
    # calculate days since birth
    days_since_birth = datetime.today() - datetime(
        user_info["year_of_birth"],
        user_info["month_of_birth"],
        user_info["day_of_birth"],
    )

    user_info["days_since_birth"] = days_since_birth.days
    return user_info


def welcome_user(user_info):
    print(f'Welcome {user_info["name"]}')
    print(f'Today is your {user_info["days_since_birth"]} day')


def calculate_physical(days):
    return math.sin((2 * math.pi / 23) * days)


def calculate_emotional(days):
    return math.sin((2 * math.pi / 28) * days)


def calculate_intellectual(days):
    return math.sin((2 * math.pi / 33) * days)


def check_if_good_or_bad_day(status, name):
    if status > 0.5:
        print(f"You have a good {name} day today")
    if status < -0.5:
        print(f"You have a bad {name} day today")
        return 1
    return 0


def check_next_day(current, next):
    if next > current:
        print("Don't worry tomorrow will be better")


def main():
    user_info = get_user_info()
    welcome_user(user_info)
    statuses = [calculate_physical, calculate_emotional, calculate_intellectual]
    names = ["physical", "emotional", "intelectual"]
    for i, status in enumerate(statuses):
        current = status(user_info["days_since_birth"])
        is_bad = check_if_good_or_bad_day(current, names[i])
        if is_bad:
            next_day = status(user_info["days_since_birth"] + 1)
            check_next_day(current, next_day)


if __name__ == "__main__":
    main()
