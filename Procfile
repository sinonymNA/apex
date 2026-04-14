api: uvicorn api.main:app --host 0.0.0.0 --port $PORT
worker: python worker/trader.py
dashboard: cd dashboard && npm install && npm run build && npx serve -s build -l $PORT
