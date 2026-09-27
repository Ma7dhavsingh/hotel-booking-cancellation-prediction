
import streamlit as st
import pandas as pd
import joblib


# Hinglish: Is line se Streamlit app ko wide screen layout me open kar rahe hain.
st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)


# Model aur preprocessor load kar rahe hain
model = joblib.load("hotel_cancellation_model_small.pkl")
preprocessor = joblib.load("hotel_cancellation_preprocessor.pkl")


model = joblib.load("hotel_cancellation_model_small.pkl")
st.write("Predict whether a hotel booking is likely to be canceled.")

st.success("Model loaded successfully!")


# Booking inputs
st.subheader("🏨 Booking Details")

# Hinglish: Is section me Hotel Type aur Lead Time ko do columns me arrange kar rahe hain, taaki UI clean aur professional dikhe.
col1, col2 = st.columns(2)

with col1:
    hotel = st.selectbox(
        "Hotel Type",
        ["City Hotel", "Resort Hotel"]
    )

with col2:
    lead_time = st.number_input(
        "Lead Time (days)",
        min_value=0,
        value=30
    )

arrival_month = st.selectbox(
    "Arrival Month",
    [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]
)

weekend_nights = st.number_input(
    "Weekend Nights",
    min_value=0,
    value=0
)

week_nights = st.number_input(
    "Week Nights",
    min_value=0,
    value=1
)

adults = st.number_input(
    "Adults",
    min_value=0,
    value=2
)

children = st.number_input(
    "Children",
    min_value=0,
    value=0
)

babies = st.number_input(
    "Babies",
    min_value=0,
    value=0
)

meal = st.selectbox(
    "Meal Type",
    ["BB", "HB", "FB", "SC", "Undefined"]
)

market_segment = st.selectbox(
    "Market Segment",
    [
        "Online TA", "Offline TA/TO", "Groups",
        "Direct", "Corporate", "Complementary",
        "Aviation", "Undefined"
    ]
)

distribution_channel = st.selectbox(
    "Distribution Channel",
    ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
)

is_repeated_guest = st.selectbox(
    "Repeated Guest",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

previous_cancellations = st.number_input(
    "Previous Cancellations",
    min_value=0,
    value=0
)

deposit_type = st.selectbox(
    "Deposit Type",
    ["No Deposit", "Non Refund", "Refundable"]
)

adr = st.number_input(
    "ADR (Average Daily Rate)",
    min_value=0.0,
    value=100.0
)

total_special_requests = st.number_input(
    "Total Special Requests",
    min_value=0,
    value=0
)

room_changed = st.selectbox(
    "Room Changed",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

customer_type = st.selectbox(
    "Customer Type",
    ["Transient", "Transient-Party", "Contract", "Group"]
)


# Derived features
total_stay_nights = weekend_nights + week_nights
total_guests = adults + children + babies

is_family = int((children > 0) or (babies > 0))
has_weekend_stay = int(weekend_nights > 0)

lead_time_category = (
    "0-7 days" if lead_time <= 7 else
    "8-30 days" if lead_time <= 30 else
    "31-90 days" if lead_time <= 90 else
    "90+ days"
)

special_request_category = (
    "No Request" if total_special_requests == 0 else
    "1-2 Requests" if total_special_requests <= 2 else
    "3+ Requests"
)

stay_duration_category = (
    "Short Stay" if total_stay_nights <= 2 else
    "Medium Stay" if total_stay_nights <= 5 else
    "Week Stay" if total_stay_nights <= 7 else
    "Long Stay"
)

booking_party_type = (
    "Solo" if total_guests == 1 else
    "Couple" if total_guests == 2 and is_family == 0 else
    "Family" if is_family == 1 else
    "Group" if total_guests >= 3 else
    "Unknown"
)

adr_category = (
    "Low" if adr <= 50 else
    "Medium" if adr <= 100 else
    "High" if adr <= 200 else
    "Very High"
)


# Default features
arrival_date_year = 2017
arrival_date_week_number = 1
arrival_date_day_of_month = 1

stays_in_weekend_nights = weekend_nights
stays_in_week_nights = week_nights

previous_bookings_not_canceled = 0
booking_changes = 0
agent = 0
company = 0
days_in_waiting_list = 0
required_car_parking_spaces = 0

reserved_room_type = "A"
assigned_room_type = "A"


# Model input DataFrame
input_data = pd.DataFrame([{
    "hotel": hotel,
    "lead_time": lead_time,
    "arrival_date_year": arrival_date_year,
    "arrival_date_month": arrival_month,
    "arrival_date_week_number": arrival_date_week_number,
    "arrival_date_day_of_month": arrival_date_day_of_month,
    "stays_in_weekend_nights": stays_in_weekend_nights,
    "stays_in_week_nights": stays_in_week_nights,
    "adults": adults,
    "children": children,
    "babies": babies,
    "meal": meal,
    "country": "Unknown",
    "market_segment": market_segment,
    "distribution_channel": distribution_channel,
    "is_repeated_guest": is_repeated_guest,
    "previous_cancellations": previous_cancellations,
    "previous_bookings_not_canceled": previous_bookings_not_canceled,
    "reserved_room_type": reserved_room_type,
    "assigned_room_type": assigned_room_type,
    "booking_changes": booking_changes,
    "deposit_type": deposit_type,
    "agent": agent,
    "company": company,
    "days_in_waiting_list": days_in_waiting_list,
    "customer_type": customer_type,
    "adr": adr,
    "required_car_parking_spaces": required_car_parking_spaces,
    "total_of_special_requests": total_special_requests,
    "total_stay_nights": total_stay_nights,
    "total_guests": total_guests,
    "is_family": is_family,
    "room_changed": room_changed,
    "has_weekend_stay": has_weekend_stay,
    "lead_time_category": lead_time_category,
    "special_request_category": special_request_category,
    "booking_party_type": booking_party_type,
    "stay_duration_category": stay_duration_category,
    "adr_category": adr_category
}])


# Prediction
if st.button("Predict Cancellation"):

    input_processed = preprocessor.transform(input_data)

    prediction = model.predict(input_processed)[0]
    probability = model.predict_proba(input_processed)[0][1]

    if prediction == 1:
        st.error("Booking is likely to be Canceled.")
    else:
        st.success("Booking is likely to be Not Canceled.")

    st.write(f"Cancellation Probability: {probability:.2%}")
