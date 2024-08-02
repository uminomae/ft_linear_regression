import json

def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def main():
    # Load theta values
    try:
        with open('theta_values.json', 'r') as f:
            theta_values = json.load(f)
        theta0 = theta_values['theta0']
        theta1 = theta_values['theta1']
    except FileNotFoundError:
        print("Model parameters not found. Please train the model first.")
        return

    # Get mileage input from user
    mileage = float(input("Enter the mileage of the car: "))

    # Estimate price
    estimated_price = estimate_price(mileage, theta0, theta1)

    print(f"Estimated price for a car with {mileage} miles: ${estimated_price:.2f}")

if __name__ == "__main__":
    main()