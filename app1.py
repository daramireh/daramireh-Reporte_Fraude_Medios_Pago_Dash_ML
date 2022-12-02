import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.express as px
import numpy as np
import dash_bootstrap_components as dbc
import pandas as pd



from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import auc, roc_auc_score, roc_curve
from dash.dependencies import Input, Output
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import svm



# cargue datos
df = pd.read_csv('df_c.csv', sep=';')


# Manipulacion de datos

# transformación para el modelo

Y = df.iloc[: , -1]
X = df.iloc[: , 0:9]

print(df.head(5))
print()

# import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 5))

names1 = ['No', 'Si']
fig = px.pie(values=df.isFraud.value_counts(), names=names1)
fig.update_layout(title="Tasa de fraude")



# dividiendo la base de datos
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=22, stratify = Y)



# selection of algorithms to consider and set performance measure
models = []
models.append(('Logistic Regression', LogisticRegression(solver='liblinear', random_state=7,
                                                         class_weight='balanced')))
models.append(('Random Forest', RandomForestClassifier(
    n_estimators=100, random_state=7)))
models.append(('SVM', SVC(gamma='auto', random_state=7)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('Decision Tree Classifier',
               DecisionTreeClassifier(random_state=7)))
models.append(('Gaussian NB', GaussianNB()))


acc_results = []
auc_results = []
names = []
# set table to table to populate with performance results
col = ['Algorithm', 'ROC AUC Mean', 'ROC AUC STD', 
       'Accuracy Mean', 'Accuracy STD']
attrition_results = pd.DataFrame(columns=col)
i = 0
# evaluate each model using cross-validation
for name, model in models:
    kfold = model_selection.KFold(
        n_splits=10)  # 10-fold cross-validation

    cv_acc_results = model_selection.cross_val_score(  # accuracy scoring
        model, X_train, y_train, cv=kfold, scoring='accuracy')

    cv_auc_results = model_selection.cross_val_score(  # roc_auc scoring
        model, X_train, y_train, cv=kfold, scoring='roc_auc')

    acc_results.append(cv_acc_results)
    auc_results.append(cv_auc_results)
    names.append(name)
    attrition_results.loc[i] = [name,
                         round(cv_auc_results.mean()*100, 2),
                         round(cv_auc_results.std()*100, 2),
                         round(cv_acc_results.mean()*100, 2),
                         round(cv_acc_results.std()*100, 2)
                         ]
    i += 1
attrition_results.sort_values(by=['ROC AUC Mean'], ascending=False)


def repetir(lista, veces):
    salida = []
    for elemento in lista:
        salida.extend([elemento] * veces)
    return salida

df = pd.DataFrame(np.concatenate((auc_results)))
df['Algoritmo']=repetir(names,10)
figbox = px.box(df, x=df.Algoritmo, y=0,
                 width=750, height=400)

def generate_table(dataframe):
    return html.Table([ 
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), 6))
        ])
    ])


# LogisticRegression
param_grid = {'C': np.arange(0.1, 2, 0.1)} # hyper-parameter list to fine-tune
log_gs = GridSearchCV(LogisticRegression(solver='liblinear', # setting GridSearchCV
                                         class_weight="balanced", 
                                         random_state=7),
                      #iid=True,
                      return_train_score=True,
                      param_grid=param_grid,
                      scoring='roc_auc',
                      cv=10)
log_grid = log_gs.fit(X_train, y_train)
log_opt = log_grid.best_estimator_

fpr, tpr, thresholds = roc_curve(y_test, log_opt.predict_proba(X_test)[:,1])
figlogreg = px.line(x=fpr, y=tpr, title='Logistic Regression')
figlogreg.update_layout(xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos")

# RandomForestClassifier
rf_classifier = RandomForestClassifier(class_weight = "balanced",
                                       random_state=7)
param_grid = {'n_estimators': [100],
              'min_samples_split':[8],
              'min_samples_leaf': [3],
              'max_depth': [15]}

rf_obj = GridSearchCV(rf_classifier,
                        #iid=True,
                        return_train_score=True,
                        param_grid=param_grid,
                        scoring='roc_auc',
                        cv=10)

grid_fit = rf_obj.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_

rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_opt.predict_proba(X_test)[:,1])
figrf = px.line(x=rf_fpr, y=rf_tpr, title='Random Forest')
figrf.update_layout(xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos")

# SMV

smv = SVC(gamma='auto', random_state=7)
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
smv_obj =  GridSearchCV(smv,
                        #iid=True,
                        param_grid=param_grid,
                        scoring='roc_auc',cv=5)
grid_fit = smv_obj.fit(X_train, y_train)
smv_opt = grid_fit.best_estimator_

smv_fpr, smv_tpr, smv_thresholds = roc_curve(y_test, smv_opt.decision_function(X_test))
figsmv = px.line(x=smv_fpr, y=smv_tpr, title='Soporte Maquina Vectorial')
figsmv.update_layout(xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos")



# KNN

knn = KNeighborsClassifier()
param_grid = {'n_neighbors': [3, 4, 5]} # hyper-parameter list to fine-tune
knn_obj =  GridSearchCV(knn,
                        #iid=True,
                        param_grid=param_grid,
                        scoring='roc_auc')
knn_fit = knn_obj.fit(X_train, y_train)
knn_opt = knn_fit.best_estimator_

knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test, knn_opt.predict_proba(X_test)[:,1])
figknn = px.line(x=knn_fpr, y=knn_tpr, title='KNN')
figknn.update_layout(xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos")


# Decision Tree Classifier

dtc = DecisionTreeClassifier(random_state=7)
param_grid = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
dtc_obj =  GridSearchCV(dtc,
                        #iid=True,
                        param_grid=param_grid,
                        scoring='roc_auc',
                        cv=5)
dtc_fit = dtc_obj.fit(X_train, y_train)
dtc_opt = dtc_fit.best_estimator_

dtc_fpr, dtc_tpr, dtc_thresholds = roc_curve(y_test, dtc_opt.predict_proba(X_test)[:,1])
figdtc = px.line(x=dtc_fpr, y=dtc_tpr, title='Decision Tree Classifier')
figdtc.update_layout(xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos")


# GaussianNB

nb_classifier = GaussianNB()

param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}
gs_NB = GridSearchCV(nb_classifier, 
                 param_grid=param_grid, 
                 cv=5,   # use any cross validation technique 
                 verbose=1, 
                 scoring='roc_auc') 
gs_NB_fit = gs_NB.fit(X_train, y_train)
gs_NB_opt = gs_NB_fit.best_estimator_

gs_fpr, gs_tpr, gs_thresholds = roc_curve(y_test, gs_NB_opt.predict_proba(X_test)[:,1])
figgs = px.line(x=gs_fpr, y=gs_tpr, title='Gaussian NB')
figgs.update_layout(xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos")


def actualizar_grafico(modelo):
    figura = ""
    if modelo == 'Logistic Regression':
        
        figura = figlogreg
        
    elif modelo == 'Random Forest':
        figura = figrf
    elif modelo == 'KNN':
        figura = figknn
    elif modelo == 'Decision Tree Classifier':
        figura = figdtc
    elif modelo == 'Gaussian NB':
        figura = figgs
    elif modelo == 'SVM':
        figura = figsmv
    return figura

def generate_table(dataframe):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), 6))
        ])
    ])


def comparate_models(model1, model2):
    log1 = ""
    log2 = ""
    if model1 == 'Logistic Regression':
        log1 = log_gs
    elif model1 == 'Random Forest':
        log1 = rf_obj
    elif model1 == 'KNN':
        log1 = knn_fit
    elif model1 == 'Decision Tree Classifier':
        log1 = dtc_fit
    elif model1 == 'Gaussian NB':
        log1 = gs_NB
    elif model1 == 'SVM':
        log1 = smv_obj

    if model2 == 'Logistic Regression':
        log2 = log_gs
    elif model2 == 'Random Forest':
        log2 = rf_obj
    elif model2 == 'KNN':
        log2 = knn_fit
    elif model2 == 'Decision Tree Classifier':
        log2 = dtc_fit
    elif model2 == 'Gaussian NB':
        log2 = gs_NB
    elif model2 == 'SVM':
        log2 = smv_obj

        log1.update_layout(xaxis_title="X Axis Title", yaxis_title="X Axis Title")

    return html.Div([
                    html.H3(model1),
                    html.H4("Mejores parametros: {}".format(log1.best_params_)),
                    html.H4("Mejor Score: {}".format(round(log1.best_score_,2))),
                    html.H3(model2),
                    html.H4("Mejores parametros: {}".format(log2.best_params_)),
                    html.H4("Mejor Score: {}".format(round(log2.best_score_,2)))
                    ], style={'columnCount': 2})

# https://www.bootstrapcdn.com/bootswatch/
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )


# Layout section: Bootstrap (https://hackerthemes.com/bootstrap-cheatsheet/)
# ************************************************************************
app.layout = html.Div(children=[
     
    html.H1('MODELO DE DETECCIÓN DE FRAUDE TRANSACCIONAL'),

    dcc.Tabs(id="tabs-graph", value='tab-1-graph', children=[
        dcc.Tab(label='Análisis Descriptivo', value='tab-1-graph'),
        dcc.Tab(label='Predicciones', value='tab-2-graph'),
    ]),

    html.Div(id='tabs-content-graph'),
   
    html.Div(id='tabs-content2-graph'),

    html.Div(id='display-selected-values')


])


@app.callback(Output('tabs-content-graph', 'children'),
                            Input('tabs-graph', 'value'))
def render_content(tab):
        if tab == 'tab-1-graph':
            return html.Div([
                dbc.Container([

                    dbc.Row(
                        dbc.Col(html.H1("Descripción transacciones",
                                        className='text-center text-primary mb-4'),
                                width=12)
                    ),

                    dbc.Row([

                        dbc.Col([
                            dcc.Dropdown(id='my-dpdn', multi=False, value='CASH_OUT',
                                        options=[{'label':x, 'value':x}
                                                for x in sorted(df['type'].unique())],
                                        ),
                            dcc.Graph(id='line-fig', figure={})
                        ],# width={'size':5, 'offset':1, 'order':1},
                        xs=12, sm=12, md=12, lg=5, xl=5
                        ),

                        dbc.Col([
                            dcc.Dropdown(id='my-dpdn2', multi=True, value=['CASH_OUT','TRANSFER'],
                                        options=[{'label':x, 'value':x}
                                                for x in sorted(df['type'].unique())],
                                        ),
                            dcc.Graph(id='line-fig2', figure={})
                        ], #width={'size':5, 'offset':0, 'order':2},
                        xs=12, sm=12, md=12, lg=5, xl=5
                        ),

                    ], justify='start'),  # Horizontal:start,center,end,between,around

                    dbc.Row([
                        dbc.Col([
                            html.P("Seleccione una transacción:",
                                style={"textDecoration": "underline"}),
                            dcc.Checklist(id='my-checklist', value=['CASH_OUT', 'TRANSFER', 'CASH_IN'],
                                        options=[{'label':x, 'value':x}
                                                for x in sorted(df['type'].unique())],
                                        labelClassName="mr-3"),
                            dcc.Graph(id='my-hist', figure={}),
                        ], #width={'size':5, 'offset':1},
                        xs=12, sm=12, md=12, lg=5, xl=5
                        ),


                    ], align="center")  # Vertical: start, center, end

                ], fluid=True)
]) 
        elif tab == 'tab-2-graph':
                    return html.Div([
                html.Div([
                    html.Div([
                    html.H3('Resultados de la evaluación promedio de los modelos supervisados'),
                    generate_table(attrition_results)
                ], style={'width': '49%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(
                        id='example-graph1',
                        figure=figbox
                    )
                ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
               
                ]), 
                    html.Div([
                    ],style={'width': '100%', 'display': 'inline-block'}),

                html.Div([
                dcc.Dropdown(
                    id='model1',
                    options=[{'label': i, 'value': i} for i in names],
                    value=names[0]
                ),
                dcc.Graph(
                        id='example-graph1',
                        figure=figlogreg
                )
                ],style={'width': '49%', 'display': 'inline-block'}),
                html.Div([
                dcc.Dropdown(
                    id='model2',
                    options=[{'label': i, 'value': i} for i in names],
                    value=names[1]
                ),
                dcc.Graph(
                        id='example-graph2',
                        figure=figrf
                )
                ],style={'width': '49%', 'display': 'inline-block', 'float': 'right'})
            ])     

    


# Callback section: connecting the components
# ************************************************************************
# Line chart - Single
@app.callback(
    Output('line-fig', 'figure'),
    Input('my-dpdn', 'value')
)
def update_graph(transaction):
    dff = df[df['type']==transaction]
    figln = px.line(dff, x='step', y='amount')
    return figln


# Line chart - multiple
@app.callback(
    Output('line-fig2', 'figure'),
    Input('my-dpdn2', 'value')
)
def update_graph(transaction):
    dff = df[df['type'].isin(transaction)]
    figln2 = px.line(dff, x='step', y='amount', color='isFraud')
    return figln2


# Histogram
@app.callback(
    Output('my-hist', 'figure'),
    Input('my-checklist', 'value')
)
def update_graph(transaction):
    dff = df[df['type'].isin(transaction)]
    dff = dff[dff['type']=='TRANSFER']
    fighist = px.bar(dff, x='type', y='isFraud')
    return fighist



@app.callback(dash.dependencies.Output('example-graph1', 'figure'),
              [dash.dependencies.Input('model1', 'value')])
def render_content(model1):
    return actualizar_grafico(model1)

@app.callback(dash.dependencies.Output('example-graph2', 'figure'),
              [dash.dependencies.Input('model2', 'value')])
def render_content(model2):
    return actualizar_grafico(model2)

@app.callback(
    Output('display-selected-values', 'children'),
    Input('tabs-graph', 'value'),
    Input('model1', 'value'),
    Input('model2', 'value')
    )
def set_display_children(tab, model1, model2):
    if tab == 'tab-2-graph':
        return comparate_models(model1, model2)




if __name__=='__main__':
    app.run_server(debug=True, port=8000)