from flask import Flask, render_template, url_for, request
import json
from database_operations import *
from initializer import *

vectorizer, feature_vector, feature_map_dict = initialise_feature_vector()
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/allrecipes')
def all_recipes():
    return render_template('allrecipe.html')


@app.route('/recipe')
def recipe():
    return render_template('recipe.html')

@app.route('/update_recipe')
def update_recipe():
    return render_template('mark_ingredients.html')

@app.route('/rank_recipes', methods=['POST'])
def rank_recipes():
    data = request.form.get("data")
    data = data.split(",")
    v2 = vectorizer.transform(data)
    v2 = np.array(np.sum(v2, axis=0))
    v2 = v2.ravel()
    scores = []
    for vector in feature_vector:
        scores.append(np.linalg.norm(vector-v2))
    res = sorted([(scores[i], i)
                  for i in range(len(scores))], reverse=True)[:50]
    ret_data = []
    for i in res:
        ret_data.append(get_food_details(food_id=feature_map_dict[i[1]])[0])
    return json.dumps(ret_data)


@app.route('/get_recipe', methods=['GET'])
def get_recipe():
    food_id = request.args.get('food_id')
    data = get_food_details(food_id=food_id, ingredients=True, procedure=True)
    return json.dumps(data[0])


@app.route('/get_all_ingredients', methods=['GET'])
def get_ingrs():
    data = get_ingredient_details(ingredient_type=False)
    data = [i[1] for i in data]
    return json.dumps(data)

@app.route('/update_type', methods=['POST'])
def get_type():
    data = get_ingredient_details()
    return json.dumps(data)

@app.route('/update_i_type', methods=['POST'])
def update_i_type():
    data = request.form.get('data')
    data = json.loads(data)
    for i_id, i_type in data:
        update_ingredient_type(i_type, i_id)
    return 1
if __name__ == "__main__":
    app.run(debug=True)
