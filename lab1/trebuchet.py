import random
import math
import matplotlib.pyplot as plt

V0 = 50  # Initial velocity of the projectile (m/s)
HEIGHT = 100  # Height of the trebuchet (m)
GRAVITY = 9.81  # Acceleration due to gravity (m/s^2)


def display_target(target):
    print("Target:", target)


def is_hit(target, bullet):
    return abs(target - bullet) <= 5


def plot_trajectory(angle):
    t = range(0, 101)
    x = [50 * math.cos(math.radians(angle)) * i for i in t]
    y = [50 * math.sin(math.radians(angle)) * i - 0.5 * 9.8 * i**2 + 100 for i in t]
    plt.plot(x, y, label=f"Trajectory (Angle: {angle}°)")
    plt.xlabel("Horizontal Distance (m)")
    plt.ylabel("Vertical Distance (m)")
    plt.title("Trebuchet Trajectory")
    # set a limit on y-axis 0
    plt.ylim(bottom=0, top=300)
    plt.xlim(left=0, right=500)

    plt.legend()
    plt.grid(True)


def trebuchet_game():
    target = random.randint(50, 340)
    display_target(target)
    shots_fired = 0
    angle = 0

    while True:
        angle = float(input("Enter the angle of the shot (in degrees): "))

        bullet = (
            V0
            * math.cos(math.radians(angle))
            * (
                V0 * math.sin(math.radians(angle))
                + math.sqrt(
                    (V0 * math.sin(math.radians(angle))) ** 2 + (2 * GRAVITY * HEIGHT)
                )
            )
            / GRAVITY
        )

        shots_fired += 1

        if is_hit(target, bullet):
            print("Congratulations! You hit the target in", shots_fired, "shots.")
            break
        else:
            print("Missed the target. Try again.")
            print("Bullet landed at:", bullet)

    plt.scatter(target, 0, color="red", label="Impact Point")
    plt.legend()

    plt.arrow(
        0,
        100,
        50 * math.cos(math.radians(angle)),
        50 * math.sin(math.radians(angle)),
        color="orange",
        width=1,
        head_width=10,
        head_length=15,
        label="Initial Velocity Vector",
    )
    plot_trajectory(angle)

    plt.show()
    plt.savefig("trebuchet.png")


if __name__ == "__main__":
    trebuchet_game()
