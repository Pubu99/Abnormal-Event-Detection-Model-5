from twilio.rest import Client

def send_alert(category: str):
    # Replace with your Twilio credentials
    account_sid = 'your_account_sid'
    auth_token = 'your_auth_token'
    client = Client(account_sid, auth_token)
    
    if category in ['Robbery', 'Shooting']:
        message = client.messages.create(
            to='+police_number',  # Police contact
            from_='+your_twilio_number',
            body=f'Alert: {category} detected! Score high.'
        )
    elif category == 'RoadAccidents':
        message = client.messages.create(
            to='+hospital_number',
            from_='+your_twilio_number',
            body=f'Alert: Accident detected! Immediate response needed.'
        )
    # Add more mappings as needed
    print(f"Alert sent for {category}")