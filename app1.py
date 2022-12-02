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

resumen = df.groupby(['step', 'type',]).agg(isFraud_sum=('isFraud', 'sum')).reset_index()

resumen['type'] = resumen['type'].astype(str)
resumen['type'] = resumen['type'].replace('0', 'TRANSFER')
resumen['type'] = resumen['type'].replace('1', 'CASH_OUT')

resumen1 = resumen.groupby(['type']).agg(isFraud_sum=('isFraud_sum', 'sum')).reset_index()



'''resumen.columns = ['_'.join(multi_index) for multi_index in resumen.columns.ravel()]
resumen = resumen.reset_index()
'''
## montos de transacciones
amount = df.groupby(['step', 'type', 'isFraud']).agg(amount_sum=('amount', 'sum')).reset_index()
'''amount.columns = ['_'.join(multi_index) for multi_index in amount.columns.ravel()]
amount = amount.reset_index()
'''
## numero de transacciones
trans = df.groupby(['step', 'type', 'isFraud']).agg(trans = ('isFraud','count')).reset_index()
'''trans.columns = ['_'.join(multi_index) for multi_index in trans.columns.ravel()]
trans = trans.reset_index()
'''
print(resumen['type'].value_counts())

# Manipulacion de datos

# transformación para el modelo

Y = df.iloc[: , -1]
X = df.iloc[: , 0:9]

'''print(df.head(5))
print()'''

# import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 5))

names1 = ['No', 'Si']

figpie = px.pie(values=df.isFraud.value_counts(), names=names1)
figpie.update_layout(title="Proporción transacciones fraude vs no fraude")

figln = px.line(amount, x='step', y='amount_sum')
figln.update_xaxes(showticklabels=True,tickangle=30,col=2)
figln.update_yaxes(title = "Valor transacciones", zeroline=True, zerolinewidth=1, zerolinecolor='#28221D')
figln.update_layout(title="Valor transacciones")

figln2 = px.line(amount, x='step', y='amount_sum', color='isFraud')
figln2.update_xaxes(showticklabels=True,tickangle=30,col=2)
figln2.update_yaxes(title = "Valor transacciones", zeroline=True, zerolinewidth=1, zerolinecolor='#28221D')
figln2.update_layout(title = 'Valor transacciones fraude vs no fraude')

figln3 = px.line(trans, x='step', y='trans', color = 'type')
figln3.update_xaxes(showticklabels=True,tickangle=30,col=2)
figln3.update_yaxes(title = "Número de transacciones", zeroline=True, zerolinewidth=1, zerolinecolor='#28221D')
figln3.update_layout(title = 'Número de transacciones')


fighist = px.bar(resumen1, x='type', y = 'isFraud_sum')
fighist.update_yaxes(title = "Número de transacciones", zeroline=True, zerolinewidth=1, zerolinecolor='#28221D')
fighist.update_layout(title = 'Número de transacciones fraudulentas por tipo de transacción')


figln4 = px.line(resumen, x='step', y='isFraud_sum', color='type')
figln4.update_xaxes(showticklabels=True,tickangle=30,col=2)
figln4.update_yaxes(title = "Número de transacciones fraudulentas", zeroline=True, zerolinewidth=1, zerolinecolor='#28221D')
figln4.update_layout(title = 'Número de transacciones fraude vs no fraude por tipo de transacción')



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
            dcc.Markdown('''
            ## Tablero para el control estadístico del riesgo transaccional

            Este reporte contine la descripción de las variables de interes, su comportamiento en el tiempo y demás información descriptiva.
            El tablero fue desarrollado en Python basado en una imagen de la versión 3.8 de Docker.
            Los modelos de Machine Learning fueron desarrollados utilizando pipelines y otras técnicas de programación que optimizan el procesamiento
            y reducen el costo computacional. A continuación se explica qué es riesgo financiero y porqué es importante el control estadístico del mismo. 

            ### ¿Qué es riesgo financiero?

            El riesgo financiero es la probabilidad de que se produzca un acontecimiento negativo que provoque pérdidas financieras en una empresa. 
            Debe de ser calculado antes de decidir llevar a cabo una inversión.
            La mayoría de empresas realizan inversiones de forma periódica para poder mantener su actividad o desarrollar nuevos proyectos 
            que le generen una fuente de ingresos.
            En cualquier inversión que se desee realizar, es fundamental cuantificar los riesgo que conlleva. En función del riesgo, 
            se decidirá finalmente si se lleva a cabo o se rechaza.
            El riesgo generalmente está ligado a la rentabilidad. Cuanto mayor es el riesgo de una inversión, 
            mayor rentabilidad se podrá obtener si sale bien. Es importante destacar que, en función del perfil de la empresa o persona que invierte, 
            se decidirán asumir un mayor o menor número de riesgos.

            Tomado de la defininición técnica de Fraude Financiero https://economipedia.com/definiciones/riesgo-financiero.html

            ### Fraude en sistema de pagos, ¿por qué este reporte estadístico?

            El fraude en pagos se produce cuando alguien roba la información de pago privada de otra persona 
            (o la engaña para que la comparta) y luego utiliza esa información para una transacción falsa o ilegal. 
            Cada vez que un nuevo método o servicio de pago gana popularidad, el panorama de los pagos cambia. Y también lo hacen los estafadores. 
            Se adaptan a cada nueva tendencia desarrollando nuevos y más sofisticados esquemas de fraude en pagos.

            Los estafadores utilizan el eslabón más débil de la cadena de acontecimientos que conducen al fraude en pagos: las personas. 
            Cualquier persona que realice pagos o utilice servicios de pago es un objetivo potencial. Por desgracia, a los delincuentes 
            no les resulta difícil manipular a las personas para conseguir sus objetivos.

            En todo el mundo, los defraudadores han adaptado, migrado y ampliado rápidamente sus tácticas de fraude aprovechándose 
            de las organizaciones y personas que no están preparadas. Si se mantienen las tendencias actuales, Juniper Research afirma 
            que las pérdidas por fraude en pagos online ascenderán a 48.000 millones de dólares en 2023. Y eso es sólo la punta del iceberg.

            Tomado de El fraude en los pagos evoluciona rápidamente. ¿Podemos estar preparados? https://www.sas.com/es_es/insights/articles/risk-fraud/payment-fraud-evolves-fast-can-we-stay-ahead.html#/

            ''', mathjax=True),
                    html.H3('Análisis de la variable de interés'),
                    html.Div([
                    dcc.Graph(
                    id='example-graph',
                    figure=figpie
                    )
                    ], style={'width': '49%', 'display': 'inline-block'}),
                    html.Div([
                        dcc.Markdown('''
                        ### Modelos de clasificación para detección de fraude transaccional

                        Los modelos de Machine Learning utilizados para determinar si una transacción es fraudulenta o no se basan en Regresión Logística, 
                        Maquina de Soporte Vectorial y Random Forest. Estos modelos son ajustados por medio de validación cruzada estratficada toda vez que 
                        la variable de respuesta (booleana que indica si hay fraude o no), se encuentra desbalanceada toda vez que las transacciones fraudulentas 
                        tienden a una menor proporción respecto de las no fraudulentas.

                        En ese orden de ideas, se presentan los resultados (scoring) de los tres modelos con la validación cruzada y se exponen 
                        los resultados de la regresión logística sin validación cruzada. Por razones de capacidad de procesamiento se tomó una muestra 
                        para la construcción del modelo y validación del algoritmo.

                        Dentro de los trabajos académicos en esta área encontramos modelos de machine learning y deep learning para predecir el 
                        fraude transaccional con medios de pagos como tarjetas de crédito y fraude en transacciones en cajeros automáticos -ATM 
                        por sus siglas en inglés.

                        Se considera un score bueno cuando está por encima del 80%
                        ''', mathjax=True)
                    ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
                    html.Div([
                    dcc.Graph(
                    id='example-graph',
                    figure=figln
                    )
                    ], style={'width': '49%', 'display': 'inline-block'}),
                    html.Div([
                    dcc.Graph(
                    id='example-graph',
                    figure=figln2
                    )
                    ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
                    html.Div([
                    dcc.Graph(
                    id='example-graph',
                    figure=figln3
                    )
                    ], style={'width': '49%', 'display': 'inline-block'}),
                    html.Div([
                        dcc.Markdown('''
                        ### Pipelines 

                            Los pipelines son herramientas que se han desarrollado para optimizar el código de la maquina de aprendizaje, 
                            permiten mejorar el rendimiento del procesamiento y reducen el costo computacional.

                            Para más información consultar la siguiente documentación de sklearn https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html''', mathjax=True)
                    ], style={'width': '100%', 'display': 'inline-block', 'float': 'right'}),
                    html.Div([
                    ],style={'width': '100%', 'display': 'inline-block'}),
                    html.Div([
                    dcc.Graph(
                    id='example-graph',
                    figure=figln4
                    )
                    ]),
                    html.Div([
                        dcc.Markdown('''
                        ### Transacciones en el tiempo
                        Se evidencia que a nivel transaccional por step, estas se han comportado homogeneamente a pesar del muestreo aleatorio
                        estas han seguido un distribución similar excepto en el step 249 y 726 donde se encuentran picos de transacciones fraudulentas.
                        ''', mathjax=True)
                    ]),
                    html.Div([
                    dcc.Graph(
                    id='example-graph',
                    figure=fighist
                    )
                    ]),
                    html.Div([
                        dcc.Markdown('''
                        ### Transacciones por tipo de transacción
                        Aquí se evidencia que la transacciones tienen un comportamiento similar, es decir, la mitad de las transacciones (ver pie)
                        muestran un comportamiento simetrico entre fraude y no fraude y esto se evidencia claramente en que las transacciones fraudulentas por
                        tipo de transacción, también se comportan de la misma forma, es decir, la mitad de ambos tipos de transacciones han sido fraude.
                        Es preciso recordar que se realizó muestreo de las variables con base en la recomendación de hacerla con submuestreo.
                        Para más información sobre el proceso de submuestreo consultar https://daramireh.github.io/Reporte_Fraude_Medios_Pago/notebooks.html
                        
                        ''', mathjax=True)
                    ]),
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