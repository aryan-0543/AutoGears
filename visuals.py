import base64
import io
import matplotlib.pyplot as plt

def create_charts(df):
    charts = {}
    category_sum = df.groupby('Category')['Amount'].sum()
    plt.figure(figsize=(6,6))
    category_sum.plot.pie(autopct='%1.1f%%')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    charts['pie'] = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    plt.figure(figsize=(8,4))
    category_sum.plot.bar(color='skyblue')
    plt.ylabel("Total Spend (₹)")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    charts['bar'] = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    return charts