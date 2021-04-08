mkdir -p ~/.streamlit
echo "[server]
headless = true
port = $PORT
enableCORS = false

[runner]
magicEnabled = false

[theme]
primaryColor='#4169e1'
backgroundColor='#FFFFFF'
secondaryBackgroundColor='#F0F2F6'
textColor='#262730'
font='sans serif'
" > ~/.streamlit/config.toml
