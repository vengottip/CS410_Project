from flask import Flask, render_template, request, jsonify
import retrieval_test
import picmodeling
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    # Retrieve JSON data sent from the frontend
    data = request.get_json()
    search_text = data.get('search_text', '')
    results_count = data.get('results_count', '')
    num = 10
    if results_count.isnumeric():
        results_count = int(results_count)
        if 1 <= results_count <= 10:
            num = results_count

    # Perform the logic for searching or displaying data here, and then return the results to the frontend
    # call the relevant model or perform database queries
    formatted_topics = picmodeling.format_lda_topics(num)
    search_result = retrieval_test.retrieve_tweets_with_query(search_text, num)
    # text_data = [['Good', 'Cybertruck', 'Model', 'evening', 'market'], ['Elon', 'Musk', 'make', 'rules', 'coin'],['Scroll', 'project', 'crypto', 'game', 'total']]
    # Example: Assume there is a variable named search_result to store the search results
    # search_result = ["Result 1", "Result 2", "Result 3"]
    chart_data = {'positive' : 1, 'negative': 2, 'neutral': 2}
    response_data = {'search_result': search_result, 'chart_data': chart_data, 'formatted_topics' : formatted_topics}
    # return json to frontend
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
