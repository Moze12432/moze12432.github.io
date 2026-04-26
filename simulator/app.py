import streamlit as st
import paho.mqtt.client as mqtt
import ssl

# MQTT details (your HiveMQ Cloud)
BROKER = "be38eefc171e4423ac4f6621a3825a5c.s1.eu.hivemq.cloud"
PORT = 8883
USERNAME = "moses"
PASSWORD = "Moses123*"

# MQTT setup
client = mqtt.Client()
client.username_pw_set(USERNAME, PASSWORD)
client.tls_set(cert_reqs=ssl.CERT_NONE)
client.tls_insecure_set(True)

client.connect(BROKER, PORT)
client.loop_start()

# Streamlit UI
st.title("📱 IoT Control Dashboard")

st.subheader("💡 LED Control")

if st.button("Turn ON"):
    client.publish("device/relay", "ON")
    st.success("LED ON")

if st.button("Turn OFF"):
    client.publish("device/relay", "OFF")
    st.warning("LED OFF")

st.subheader("🌡️ Temperature Feed")
st.write("Check MQTT client to see live updates")
