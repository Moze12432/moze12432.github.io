import streamlit as st
import paho.mqtt.client as mqtt
import ssl

BROKER = "be38eefc171e4423ac4f6621a3825a5c.s1.eu.hivemq.cloud"
PORT = 8883
USERNAME = "moses"
PASSWORD = "Moses123*"

# Store state
if "led_status" not in st.session_state:
    st.session_state.led_status = "OFF"

if "temperature" not in st.session_state:
    st.session_state.temperature = "--"

# MQTT setup
client = mqtt.Client()
client.username_pw_set(USERNAME, PASSWORD)
client.tls_set(cert_reqs=ssl.CERT_NONE)
client.tls_insecure_set(True)

# When message received
def on_message(client, userdata, msg):
    message = msg.payload.decode()

    if msg.topic == "device/relay":
        st.session_state.led_status = message

    if msg.topic == "sensor/temperature":
        st.session_state.temperature = message

client.on_message = on_message

client.connect(BROKER, PORT)
client.subscribe("device/relay")
client.subscribe("sensor/temperature")
client.loop_start()

# UI
st.title("📱 IoT Dashboard")

# LED status
st.subheader("💡 LED Status")
if st.session_state.led_status == "ON":
    st.success("LED is ON")
else:
    st.error("LED is OFF")

# Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Turn ON"):
        client.publish("device/relay", "ON")

with col2:
    if st.button("Turn OFF"):
        client.publish("device/relay", "OFF")

# Temperature
st.subheader("🌡️ Temperature")
st.metric(label="Current Temp", value=st.session_state.temperature)
