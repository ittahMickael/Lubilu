
from flask import Flask, jsonify, render_template, request
from sklearn.externals import joblib
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt





pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('precision', 0)

app = Flask(__name__, static_url_path='')

#index.html
@app.route('/')
def root():
        print('Nouvelle connexion : ', time.localtime())
        return render_template('index.html')


#prediction : resultat_predict.html
@app.route('/result', methods=['GET', 'POST'])
def predict():
        result = request.form
        carosserie=result['carosserie']
        alimentation=result['alimentation']
        controleDS=result['controleDS']
        marque=result['marque']
        pta=result['pta']
        vitessemax=result['vitessemax']
        longueurmm=float(result['longueurmm'])
        
        mail=result['email']
        
        df1= pd.DataFrame([[controleDS,marque,pta,vitessemax,longueurmm,carosserie,alimentation]],columns=['Controle_dynamique_de_stabilite','Marque', 'PTAC','Vitesse_Maxi','QLNVLO', 'Carosserie','Alimentation'])
        x1 = pd.get_dummies(df1)
        


        df_type = pd.DataFrame( columns=['QLNVLO', 'Controle_dynamique_de_stabilite_N',
       'Controle_dynamique_de_stabilite_NR',
       'Controle_dynamique_de_stabilite_O',
       'Controle_dynamique_de_stabilite_S', 'Marque_AUTRE', 'Marque_Audi',
       'Marque_BMW', 'Marque_Citroen', 'Marque_Fiat', 'Marque_Ford',
       'Marque_Mercedes', 'Marque_NR', 'Marque_Opel', 'Marque_Peugeot',
       'Marque_Renault', 'Marque_Volkswagen', 'PTAC_(1.41e+03,1.53e+03]',
       'PTAC_(1.53e+03,1.62e+03]', 'PTAC_(1.62e+03,1.69e+03]',
       'PTAC_(1.69e+03,1.78e+03]', 'PTAC_(1.78e+03,1.85e+03]',
       'PTAC_(1.85e+03,1.94e+03]', 'PTAC_(1.94e+03,2.06e+03]',
       'PTAC_(2.06e+03,2.32e+03]', 'PTAC_(2.32e+03,3.5e+03]',
       'PTAC_[895,1.41e+03]', 'Vitesse_Maxi_(150,160]',
       'Vitesse_Maxi_(160,163]', 'Vitesse_Maxi_(163,170]',
       'Vitesse_Maxi_(170,175]', 'Vitesse_Maxi_(175,180]',
       'Vitesse_Maxi_(180,185]', 'Vitesse_Maxi_(185,191]',
       'Vitesse_Maxi_(191,203]', 'Vitesse_Maxi_(203,329]',
       'Vitesse_Maxi_[60,150]', 'Carosserie_Autres', 'Carosserie_Berline',
       'Carosserie_Break', 'Carosserie_CCI', 'Carosserie_Cabriolet/Coupe',
       'Carosserie_Espace', 'Carosserie_Fourgon', 'Carosserie_Fourgonnette',
       'Carosserie_TousTerrains', 'Alimentation_Autres', 'Alimentation_Diesel',
       'Alimentation_Essence'])
       
        features = ['QLNVLO', 'Controle_dynamique_de_stabilite_N',
       'Controle_dynamique_de_stabilite_NR',
       'Controle_dynamique_de_stabilite_O',
       'Controle_dynamique_de_stabilite_S', 'Marque_AUTRE', 'Marque_Audi',
       'Marque_BMW', 'Marque_Citroen', 'Marque_Fiat', 'Marque_Ford',
       'Marque_Mercedes', 'Marque_NR', 'Marque_Opel', 'Marque_Peugeot',
       'Marque_Renault', 'Marque_Volkswagen', 'PTAC_(1.41e+03,1.53e+03]',
       'PTAC_(1.53e+03,1.62e+03]', 'PTAC_(1.62e+03,1.69e+03]',
       'PTAC_(1.69e+03,1.78e+03]', 'PTAC_(1.78e+03,1.85e+03]',
       'PTAC_(1.85e+03,1.94e+03]', 'PTAC_(1.94e+03,2.06e+03]',
       'PTAC_(2.06e+03,2.32e+03]', 'PTAC_(2.32e+03,3.5e+03]',
       'PTAC_[895,1.41e+03]', 'Vitesse_Maxi_(150,160]',
       'Vitesse_Maxi_(160,163]', 'Vitesse_Maxi_(163,170]',
       'Vitesse_Maxi_(170,175]', 'Vitesse_Maxi_(175,180]',
       'Vitesse_Maxi_(180,185]', 'Vitesse_Maxi_(185,191]',
       'Vitesse_Maxi_(191,203]', 'Vitesse_Maxi_(203,329]',
       'Vitesse_Maxi_[60,150]', 'Carosserie_Autres', 'Carosserie_Berline',
       'Carosserie_Break', 'Carosserie_CCI', 'Carosserie_Cabriolet/Coupe',
       'Carosserie_Espace', 'Carosserie_Fourgon', 'Carosserie_Fourgonnette',
       'Carosserie_TousTerrains', 'Alimentation_Autres', 'Alimentation_Diesel',
       'Alimentation_Essence']

        # Usage
        dfs = [df_type,x1]
        full_df = concat_ordered_columns(dfs)
        full_df= full_df.replace(np.nan, 0)

        ##plot
        importances = clf.feature_importances_
        indices = np.argsort(importances)
        plt.figure(1, figsize=(10,20))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.savefig('static/images/importance_plot.png', bbox_inches='tight')
        plt.close()

        prediction = round(float(clf.predict(full_df)),1)

        return render_template('resultat_predict.html', result=prediction)

     

 
#API        
@app.route('/api/predict/query/login=<user_id>/carosserie=<carosserie>&alimentation=<alimentation>&controleDS=<controleDS>&marque=<marque>&pta=<pta>&vitessemax=<vitessemax>&longueurmm=<longueurmm>')
def predictCM(carosserie, alimentation, controleDS, marque, pta, vitessemax, longueurmm,user_id ):
        longueurmm=int(str(longueurmm))
        df1= pd.DataFrame([[controleDS,marque,pta,vitessemax,longueurmm,carosserie,alimentation]],columns=['Controle_dynamique_de_stabilite','Marque', 'PTAC','Vitesse_Maxi','QLNVLO', 'Carosserie','Alimentation'])
        x1 = pd.get_dummies(df1)
        df_type = pd.DataFrame( columns=['QLNVLO', 'Controle_dynamique_de_stabilite_N',
       'Controle_dynamique_de_stabilite_NR',
       'Controle_dynamique_de_stabilite_O',
       'Controle_dynamique_de_stabilite_S', 'Marque_AUTRE', 'Marque_Audi',
       'Marque_BMW', 'Marque_Citroen', 'Marque_Fiat', 'Marque_Ford',
       'Marque_Mercedes', 'Marque_NR', 'Marque_Opel', 'Marque_Peugeot',
       'Marque_Renault', 'Marque_Volkswagen', 'PTAC_(1.41e+03,1.53e+03]',
       'PTAC_(1.53e+03,1.62e+03]', 'PTAC_(1.62e+03,1.69e+03]',
       'PTAC_(1.69e+03,1.78e+03]', 'PTAC_(1.78e+03,1.85e+03]',
       'PTAC_(1.85e+03,1.94e+03]', 'PTAC_(1.94e+03,2.06e+03]',
       'PTAC_(2.06e+03,2.32e+03]', 'PTAC_(2.32e+03,3.5e+03]',
       'PTAC_[895,1.41e+03]', 'Vitesse_Maxi_(150,160]',
       'Vitesse_Maxi_(160,163]', 'Vitesse_Maxi_(163,170]',
       'Vitesse_Maxi_(170,175]', 'Vitesse_Maxi_(175,180]',
       'Vitesse_Maxi_(180,185]', 'Vitesse_Maxi_(185,191]',
       'Vitesse_Maxi_(191,203]', 'Vitesse_Maxi_(203,329]',
       'Vitesse_Maxi_[60,150]', 'Carosserie_Autres', 'Carosserie_Berline',
       'Carosserie_Break', 'Carosserie_CCI', 'Carosserie_Cabriolet/Coupe',
       'Carosserie_Espace', 'Carosserie_Fourgon', 'Carosserie_Fourgonnette',
       'Carosserie_TousTerrains', 'Alimentation_Autres', 'Alimentation_Diesel',
       'Alimentation_Essence'])

        # Usage
        dfs = [df_type,x1]
        full_df = concat_ordered_columns(dfs)
        full_df= full_df.replace(np.nan, 0)
        
        prediction = round(float(clf.predict(full_df)),1)
        return str(prediction)

#Page d'erreur       
@app.errorhandler(404)
def page_not_found(error):
   return render_template('404.html', title = '404'), 404


#Autres fonctions

##Concatenation
def concat_ordered_columns(frames):
    columns_ordered = []
    for frame in frames:
        columns_ordered.extend(x for x in frame.columns if x not in columns_ordered)
    final_df = pd.concat(frames)    
    return final_df[columns_ordered]  


if __name__ == '__main__':
        clf = joblib.load('RandomForestModel/model.pkl')
        app.run(host='0.0.0.0')
