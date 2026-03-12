import smtplib

# Replace with your details
EMAIL = "parlapallisomasekhar@gmail.com"
APP_PASSWORD = "bmcgqbieduypaqtx" # Using your current password

print(f"Attempting to connect to smtp.gmail.com:587 for {EMAIL}...")
try:
    server = smtplib.SMTP("smtp.gmail.com", 587, timeout=10)
    server.set_debuglevel(1) # See the raw conversation
    server.starttls()
    server.login(EMAIL, APP_PASSWORD)
    print("\n✅ SMTP Connected Successfully!")
    server.quit()
except Exception as e:
    print(f"\n❌ SMTP Error: {e}")
    print("\nSuggestions:")
    print("1. Ensure 2-Step Verification is ON in Google Security.")
    print("2. Ensure you are using a 16-character 'App Password', not your login password.")
    print("3. Check if your network/firewall is blocking Port 587.")
