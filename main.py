from flask import Flask,render_template,request
app = Flask(__name__)
import pickle

file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/',methods=['GET','POST'])
def hello_world():
    if request.method == "POST":
        print(request.form)
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        pain = int(myDict['pain'])
        runnyNose = int(myDict['runnyNose'])
        diffBreath = int(myDict['diffBreath'])

        # code for inference
        input_features = [fever,pain,age,runnyNose,diffBreath]
        infprob = clf.predict_proba([input_features])[0][1]
        print(round(infprob))
        return render_template('show.html',inf=round(infprob*100))
    return render_template('index.html')
    #return "Hello World!" + str(infprob)


if __name__ == "__main__":
    app.run(debug=True)
       