#!/usr/bin/env python3
"""
Lariat Bible API Server
Exposes recipe and ingredient data through REST API endpoints
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from data_importers import UnifiedImporter, ExcelImporter
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize importers
unified_importer = UnifiedImporter()
excel_importer = unified_importer.excel_importer

# Cache for imported data (reload on demand)
_data_cache = {
    'recipes': None,
    'ingredients': None,
    'last_updated': None
}


def reload_data():
    """Reload all data from files"""
    global _data_cache

    logger.info("Reloading data from files...")

    # Import recipes
    recipes_data = excel_importer.import_menu_bible_recipes()
    _data_cache['recipes'] = recipes_data

    # Import ingredients
    ingredients_data = excel_importer.import_ingredient_database()
    _data_cache['ingredients'] = ingredients_data

    _data_cache['last_updated'] = datetime.now().isoformat()

    logger.info(f"Data reloaded: {len(recipes_data['recipes']) if recipes_data else 0} recipes, "
                f"{len(ingredients_data) if ingredients_data else 0} ingredients")


# Load data on startup
reload_data()


@app.route('/')
def index():
    """API root endpoint"""
    return jsonify({
        'name': 'Lariat Bible API',
        'version': '1.0',
        'endpoints': {
            '/api/recipes': 'Get all recipes',
            '/api/recipes/<name>': 'Get specific recipe',
            '/api/recipes/category/<category>': 'Get recipes by category',
            '/api/ingredients': 'Get all ingredients',
            '/api/ingredients/search': 'Search ingredients',
            '/api/ingredients/category/<category>': 'Get ingredients by category',
            '/api/stats': 'Get statistics',
            '/api/reload': 'Reload data from files'
        }
    })


@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'data_loaded': _data_cache['recipes'] is not None,
        'last_updated': _data_cache['last_updated']
    })


@app.route('/api/recipes')
def get_all_recipes():
    """Get all recipes"""
    if not _data_cache['recipes']:
        return jsonify({'error': 'No recipes loaded'}), 404

    recipes = _data_cache['recipes']['recipes']

    # Option to include ingredients or just summary
    include_ingredients = request.args.get('include_ingredients', 'true').lower() == 'true'

    if not include_ingredients:
        # Return summary only
        recipes_summary = [
            {
                'name': r['name'],
                'category': r['category'],
                'yield': r['base_yield'],
                'ingredient_count': len(r['ingredients'])
            }
            for r in recipes
        ]
        return jsonify({
            'recipes': recipes_summary,
            'count': len(recipes_summary)
        })

    return jsonify({
        'recipes': recipes,
        'count': len(recipes)
    })


@app.route('/api/recipes/<recipe_name>')
def get_recipe(recipe_name):
    """Get specific recipe by name"""
    if not _data_cache['recipes']:
        return jsonify({'error': 'No recipes loaded'}), 404

    recipes = _data_cache['recipes']['recipes']

    # Find recipe (case-insensitive)
    recipe = next(
        (r for r in recipes if r['name'].lower() == recipe_name.lower()),
        None
    )

    if not recipe:
        return jsonify({'error': f'Recipe "{recipe_name}" not found'}), 404

    return jsonify(recipe)


@app.route('/api/recipes/category/<category>')
def get_recipes_by_category(category):
    """Get recipes by category"""
    if not _data_cache['recipes']:
        return jsonify({'error': 'No recipes loaded'}), 404

    recipes = _data_cache['recipes']['recipes']

    # Filter by category (case-insensitive)
    filtered = [
        r for r in recipes
        if r.get('category', '').lower() == category.lower()
    ]

    return jsonify({
        'category': category,
        'recipes': filtered,
        'count': len(filtered)
    })


@app.route('/api/categories')
def get_categories():
    """Get all recipe categories"""
    if not _data_cache['recipes']:
        return jsonify({'error': 'No recipes loaded'}), 404

    recipes = _data_cache['recipes']['recipes']

    # Get unique categories with counts
    categories = {}
    for recipe in recipes:
        cat = recipe.get('category', 'Unknown')
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1

    return jsonify({
        'categories': [
            {'name': cat, 'count': count}
            for cat, count in sorted(categories.items())
        ],
        'total': len(categories)
    })


@app.route('/api/ingredients')
def get_all_ingredients():
    """Get all ingredients"""
    if not _data_cache['ingredients']:
        return jsonify({'error': 'No ingredients loaded'}), 404

    ingredients = _data_cache['ingredients']

    return jsonify({
        'ingredients': ingredients,
        'count': len(ingredients)
    })


@app.route('/api/ingredients/search')
def search_ingredients():
    """Search ingredients by name"""
    if not _data_cache['ingredients']:
        return jsonify({'error': 'No ingredients loaded'}), 404

    query = request.args.get('q', '').lower()
    if not query:
        return jsonify({'error': 'Missing search query parameter "q"'}), 400

    ingredients = _data_cache['ingredients']

    # Search by name
    results = [
        ing for ing in ingredients
        if query in ing['name'].lower()
    ]

    return jsonify({
        'query': query,
        'results': results,
        'count': len(results)
    })


@app.route('/api/ingredients/category/<category>')
def get_ingredients_by_category(category):
    """Get ingredients by category"""
    if not _data_cache['ingredients']:
        return jsonify({'error': 'No ingredients loaded'}), 404

    ingredients = _data_cache['ingredients']

    # Filter by category (case-insensitive)
    filtered = [
        ing for ing in ingredients
        if ing.get('category', '').lower() == category.lower()
    ]

    return jsonify({
        'category': category,
        'ingredients': filtered,
        'count': len(filtered)
    })


@app.route('/api/ingredients/expensive')
def get_expensive_ingredients():
    """Get most expensive ingredients"""
    if not _data_cache['ingredients']:
        return jsonify({'error': 'No ingredients loaded'}), 404

    ingredients = _data_cache['ingredients']
    limit = request.args.get('limit', 10, type=int)

    # Sort by cost per lb
    sorted_ingredients = sorted(
        [ing for ing in ingredients if ing.get('cost_per_lb', 0) > 0],
        key=lambda x: x.get('cost_per_lb', 0),
        reverse=True
    )

    return jsonify({
        'ingredients': sorted_ingredients[:limit],
        'count': len(sorted_ingredients[:limit])
    })


@app.route('/api/stats')
def get_stats():
    """Get overall statistics"""
    stats = {
        'recipes': {
            'total': 0,
            'by_category': {}
        },
        'ingredients': {
            'total': 0,
            'by_category': {},
            'average_cost_per_lb': 0
        },
        'last_updated': _data_cache['last_updated']
    }

    # Recipe stats
    if _data_cache['recipes']:
        recipes = _data_cache['recipes']['recipes']
        stats['recipes']['total'] = len(recipes)

        for recipe in recipes:
            cat = recipe.get('category', 'Unknown')
            if cat not in stats['recipes']['by_category']:
                stats['recipes']['by_category'][cat] = 0
            stats['recipes']['by_category'][cat] += 1

    # Ingredient stats
    if _data_cache['ingredients']:
        ingredients = _data_cache['ingredients']
        stats['ingredients']['total'] = len(ingredients)

        costs = []
        for ing in ingredients:
            cat = ing.get('category', 'Unknown')
            if cat not in stats['ingredients']['by_category']:
                stats['ingredients']['by_category'][cat] = 0
            stats['ingredients']['by_category'][cat] += 1

            if ing.get('cost_per_lb'):
                costs.append(ing['cost_per_lb'])

        if costs:
            stats['ingredients']['average_cost_per_lb'] = sum(costs) / len(costs)

    return jsonify(stats)


@app.route('/api/reload', methods=['POST'])
def reload_endpoint():
    """Reload data from files"""
    try:
        reload_data()
        return jsonify({
            'status': 'success',
            'message': 'Data reloaded successfully',
            'timestamp': _data_cache['last_updated']
        })
    except Exception as e:
        logger.error(f"Error reloading data: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))

    print("=" * 80)
    print("LARIAT BIBLE API SERVER")
    print("=" * 80)
    print(f"\nStarting server on http://localhost:{port}")
    print(f"\nAPI Endpoints:")
    print(f"  • http://localhost:{port}/")
    print(f"  • http://localhost:{port}/api/recipes")
    print(f"  • http://localhost:{port}/api/ingredients")
    print(f"  • http://localhost:{port}/api/stats")
    print("\n" + "=" * 80 + "\n")

    app.run(debug=True, port=port, host='0.0.0.0')
