from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import json
import os

app = Flask(__name__)

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# -------------------------------
# DATA GENERATION
# -------------------------------

def generate_data():
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        'tenure': np.random.randint(0, 72, n),
        'monthly_charges': np.random.uniform(20, 120, n),
        'contract': np.random.choice(
            ['Month-to-month', 'One year', 'Two year'],
            n,
            p=[0.5, 0.3, 0.2]
        ),
        'support_tickets': np.random.poisson(3, n),
        'age': np.random.randint(18, 70, n)
    })

    df['total_charges'] = df['monthly_charges'] * df['tenure']

    churn = []

    for i in range(n):
        prob = 0.1

        if df.loc[i, 'contract'] == 'Month-to-month':
            prob += 0.3
        if df.loc[i, 'tenure'] < 6:
            prob += 0.2
        if df.loc[i, 'monthly_charges'] > 80:
            prob += 0.15
        if df.loc[i, 'support_tickets'] > 5:
            prob += 0.2
        if df.loc[i, 'contract'] == 'Two year':
            prob -= 0.2
        if df.loc[i, 'tenure'] > 24:
            prob -= 0.15

        prob = max(0, min(1, prob))
        churn.append(1 if np.random.random() < prob else 0)

    df['churned'] = churn
    return df


df = generate_data()

# -------------------------------
# MODEL TRAINING
# -------------------------------

def train_churn_model(dataframe):
    df_model = dataframe.copy()

    contract_map = {
        'Month-to-month': 0,
        'One year': 1,
        'Two year': 2
    }

    df_model['contract_encoded'] = df_model['contract'].map(contract_map)

    features = [
        'tenure',
        'monthly_charges',
        'total_charges',
        'support_tickets',
        'age',
        'contract_encoded'
    ]

    X = df_model[features]
    y = df_model['churned']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    return model, accuracy, cm, features


model, accuracy, confusion_mat, feature_columns = train_churn_model(df)

# -------------------------------
# ROUTES
# -------------------------------

@app.route('/')
def home():
    total_customers = len(df)
    churned_customers = df['churned'].sum()
    churn_rate = (churned_customers / total_customers) * 100
    avg_tenure = df['tenure'].mean()

    churn_by_contract = df.groupby('contract')['churned'].mean() * 100

    # Graph 1 - Churn Rate by Contract Type
    fig1 = go.Figure(go.Bar(
        x=churn_by_contract.index,
        y=churn_by_contract.values,
        marker_color=['#ef5350', '#ffa726', '#66bb6a'],
        text=[f'{val:.1f}%' for val in churn_by_contract.values],
        textposition='outside'
    ))
    fig1.update_layout(
        title='Churn Rate by Contract Type',
        xaxis_title='Contract Type',
        yaxis_title='Churn Rate (%)',
        template='plotly_white',
        height=300,
        margin=dict(t=50, b=50, l=50, r=20)
    )

    # Graph 2 - Tenure Distribution
    fig2 = px.histogram(
        df,
        x='tenure',
        color='churned',
        barmode='overlay',
        title='Tenure Distribution by Churn',
        labels={'churned': 'Churned', 'tenure': 'Tenure (months)'},
        template='plotly_white',
        color_discrete_map={0: '#4db8ff', 1: '#ff6666'}
    )
    fig2.update_layout(
        height=300,
        margin=dict(t=50, b=50, l=50, r=20),
        legend=dict(title='Churned', orientation='h', y=1.1, x=0.7)
    )

    # Graph 3 - Monthly Charges vs Churn
    fig3 = px.box(
        df,
        x='churned',
        y='monthly_charges',
        title='Monthly Charges vs Churn',
        labels={'churned': 'Churned', 'monthly_charges': 'Monthly Charges ($)'},
        template='plotly_white',
        color='churned',
        color_discrete_map={0: '#4db8ff', 1: '#ff6666'}
    )
    fig3.update_layout(
        height=300,
        margin=dict(t=50, b=50, l=50, r=20),
        showlegend=False
    )
    fig3.update_xaxes(ticktext=['No', 'Yes'], tickvals=[0, 1])

    # Graph 4 - Feature Importance
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=True)

    fig4 = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title='Feature Importance',
        labels={'importance': 'Importance', 'feature': 'Feature'},
        template='plotly_white',
        color='importance',
        color_continuous_scale='viridis'
    )
    fig4.update_layout(
        height=300,
        margin=dict(t=50, b=50, l=50, r=20),
        showlegend=False
    )

    # Graph 5 - Confusion Matrix
    fig5 = px.imshow(
        confusion_mat,
        text_auto=True,
        title='Confusion Matrix',
        labels=dict(x="Predicted", y="Actual"),
        x=['No Churn', 'Churn'],
        y=['No Churn', 'Churn'],
        template='plotly_white',
        color_continuous_scale='Blues'
    )
    fig5.update_layout(
        height=350,
        margin=dict(t=50, b=50, l=50, r=50)
    )

    # Convert graphs to JSON with proper numpy handling
    graph1_json = json.loads(json.dumps(fig1.to_dict(), cls=NumpyEncoder))
    graph2_json = json.loads(json.dumps(fig2.to_dict(), cls=NumpyEncoder))
    graph3_json = json.loads(json.dumps(fig3.to_dict(), cls=NumpyEncoder))
    graph4_json = json.loads(json.dumps(fig4.to_dict(), cls=NumpyEncoder))
    graph5_json = json.loads(json.dumps(fig5.to_dict(), cls=NumpyEncoder))

    return render_template(
        'index.html',
        total_customers=total_customers,
        churned_customers=churned_customers,
        churn_rate=f'{churn_rate:.1f}',
        avg_tenure=f'{avg_tenure:.1f}',
        model_accuracy=f'{accuracy*100:.1f}',
        graph1_json=graph1_json,
        graph2_json=graph2_json,
        graph3_json=graph3_json,
        graph4_json=graph4_json,
        graph5_json=graph5_json
    )


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    contract_map = {
        'Month-to-month': 0,
        'One year': 1,
        'Two year': 2
    }

    input_df = pd.DataFrame([{
        'tenure': float(data['tenure']),
        'monthly_charges': float(data['monthly_charges']),
        'total_charges': float(data['tenure']) * float(data['monthly_charges']),
        'support_tickets': int(data['support_tickets']),
        'age': int(data['age']),
        'contract_encoded': contract_map[data['contract']]
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return jsonify({
        'will_churn': bool(prediction),
        'churn_probability': f'{probability*100:.1f}',
        'risk_level': (
            'High' if probability > 0.7
            else 'Medium' if probability > 0.4
            else 'Low'
        )
    })


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )
