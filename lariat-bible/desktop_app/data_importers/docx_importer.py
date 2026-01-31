"""
DOCX Recipe Importer
Import recipes from Lariat Recipe Book.docx
"""

from docx import Document
from typing import List, Dict, Optional
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocxRecipeImporter:
    """Import recipes from Word documents"""

    def __init__(self):
        self.recipes = []

    def import_recipe_book(self, docx_path: str) -> List[Dict]:
        """
        Import all recipes from Lariat Recipe Book

        Args:
            docx_path: Path to .docx file

        Returns:
            List of recipe dictionaries
        """
        doc = Document(docx_path)

        # Get all paragraphs as list
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        recipes = []
        i = 0

        while i < len(paragraphs):
            text = paragraphs[i]

            # Check if this could be a recipe title
            if self._could_be_recipe_title(text, paragraphs, i):
                recipe = self._parse_recipe(paragraphs, i)
                if recipe and recipe.get('ingredients'):
                    recipes.append(recipe)
                    logger.info(f"Imported recipe: {recipe['name']} ({len(recipe['ingredients'])} ingredients, {len(recipe['instructions'])} instructions)")
                    # Skip to end of this recipe
                    i = recipe.get('_end_index', i + 1)
                else:
                    i += 1
            else:
                i += 1

        logger.info(f"Imported {len(recipes)} recipes from {docx_path}")
        self.recipes = recipes
        return recipes

    def _could_be_recipe_title(self, text: str, paragraphs: List[str], index: int) -> bool:
        """Check if text could be a recipe title by looking ahead"""
        # Skip known non-titles
        if text.lower() in ['lariat recipe book', 'ingredients', 'ingredients:',
                           'directions', 'directions:', 'procedure', 'procedure:',
                           'process', 'process:', 'instructions', 'instructions:',
                           'base', 'southwest flavor layer']:
            return False

        # Skip if starts with number, measurement, or fraction (it's an ingredient!)
        if re.match(r'^\d+', text) or text.startswith('-') or text.startswith('•'):
            return False
        if re.match(r'^[\u00BC-\u00BE\u2150-\u215E]', text):
            return False

        # Skip if starts with measurement words (it's an ingredient!)
        measurements = ['cup', 'tbsp', 'tsp', 'lb', 'oz', 'gallon', 'quart', 'qt', 'ea',
                       'kg', 'gram', 'liter', 'ml', 'bag', 'bunch', 'case', 'stick', 'box']
        first_word = text.split()[0].lower() if text.split() else ''
        if any(first_word.startswith(m) for m in measurements):
            return False

        # Skip if it looks like an ingredient pattern (has measurement in first few words)
        # Example: "Buttermilk - ½ Gallon Buttermilk" or "Garlic Powder - ¼ cup"
        if re.search(r'[\d\u00BC-\u00BE\u2150-\u215E]', text[:20]):
            return False

        # Skip yield lines
        if text.lower().startswith('(yields') or text.lower().startswith('yield') or 'yield' in text.lower()[:30]:
            return False

        # Skip "for the" section headers
        if text.lower().startswith('for the') or text.lower().startswith('for '):
            return False

        # Skip if ALL CAPS (it's an ingredient or instruction)
        if text.isupper():
            return False

        # Skip if too long
        if len(text.split()) > 10:
            return False

        # Look ahead to see if next few lines contain ingredients
        ingredient_count = 0
        for j in range(index + 1, min(index + 10, len(paragraphs))):
            next_text = paragraphs[j]
            if self._is_ingredient_line(next_text):
                ingredient_count += 1
            elif next_text.lower() in ['ingredients', 'ingredients:', 'base', 'southwest flavor layer']:
                continue  # Skip section headers
            elif next_text.lower().startswith('(yields') or 'yield' in next_text.lower():
                continue  # Skip yields
            elif ingredient_count > 0:
                break  # Stop if we found ingredients then hit something else

        # If we found at least 3 ingredients in the next few lines, this is probably a title
        return ingredient_count >= 3

    def _parse_recipe(self, paragraphs: List[str], start_index: int) -> Optional[Dict]:
        """Parse a complete recipe starting at the given index"""
        recipe = {
            'name': paragraphs[start_index],
            'ingredients': [],
            'instructions': [],
            'yield': None,
            'category': self._categorize_recipe(paragraphs[start_index]),
            '_end_index': start_index + 1
        }

        current_section = None
        i = start_index + 1

        while i < len(paragraphs):
            text = paragraphs[i]
            text_lower = text.lower()

            # Check if we've hit the next recipe
            if self._could_be_recipe_title(text, paragraphs, i):
                break

            # Detect yield
            if 'yield' in text_lower:
                recipe['yield'] = text
                i += 1
                continue

            # Detect section headers
            if text_lower in ['ingredients', 'ingredients:', 'base', 'southwest flavor layer']:
                current_section = 'ingredients'
                i += 1
                continue
            elif text_lower in ['procedure', 'procedure:', 'directions', 'directions:',
                               'process', 'process:', 'instructions', 'instructions:']:
                current_section = 'instructions'
                i += 1
                continue

            # Process ingredients
            if current_section == 'ingredients' or (current_section is None and self._is_ingredient_line(text)):
                if self._is_ingredient_line(text):
                    current_section = 'ingredients'  # Auto-detect ingredients section
                    recipe['ingredients'].append(self._parse_ingredient(text))

            # Process instructions
            elif current_section == 'instructions':
                if text and not text_lower.startswith('yield'):
                    recipe['instructions'].append(text)

            # If we haven't found ingredients yet and this doesn't look like an ingredient,
            # it might be yield info or we need to keep looking
            elif current_section is None:
                if not self._is_ingredient_line(text):
                    # Maybe yield or other metadata
                    pass

            i += 1
            recipe['_end_index'] = i

        return recipe

    def _is_ingredient_line(self, text: str) -> bool:
        """Check if text is an ingredient line"""
        # Skip section headers
        if text.lower() in ['ingredients', 'ingredients:', 'directions', 'directions:',
                           'procedure', 'procedure:', 'process', 'process:',
                           'instructions', 'instructions:', 'base', 'southwest flavor layer',
                           'for cream cheese spread:', 'for garlic spread:',
                           'for grilled three-cheese sandwich:', 'for the dressing:',
                           'for the slaw']:
            return False

        # Ingredient patterns: starts with number, measurement, fraction, or bullet
        if re.match(r'^\d+', text):
            return True
        if text.startswith('-') or text.startswith('•'):
            return True

        # Check for fraction patterns at start (½, ¼, ⅓, etc.)
        if re.match(r'^[\u00BC-\u00BE\u2150-\u215E]', text):
            return True

        # Common measurement words at start
        measurements = ['cup', 'tbsp', 'tsp', 'lb', 'oz', 'gallon', 'quart', 'qt', 'ea',
                       'kg', 'gram', 'liter', 'ml', 'bag', 'bunch', 'case', 'stick', 'box']
        first_word = text.split()[0].lower() if text.split() else ''
        if any(first_word.startswith(m) for m in measurements):
            return True

        # Uppercase ingredients (e.g., "HEAVY CREAM", "WHOLE MILK")
        if text.isupper() and 1 <= len(text.split()) <= 8:
            return True

        # Check for measurement patterns anywhere in first part
        # Examples: "Buttermilk - ½ Gallon Buttermilk"
        if re.search(r'[\d\u00BC-\u00BE\u2150-\u215E]', text[:30]):
            return True

        return False

    def _parse_ingredient(self, text: str) -> Dict:
        """Parse ingredient line into structured data"""
        # Remove leading bullet or dash
        text = text.lstrip('•- ').strip()

        # Try to extract quantity, unit, and name
        # Pattern: number/fraction + unit + name
        match = re.match(r'^([\d\s\/\.\-\u00BC-\u00BE\u2150-\u215E]+)\s+([a-zA-Z\.\#]+)\s+(.+)$', text)

        if match:
            return {
                'quantity': match.group(1).strip(),
                'unit': match.group(2).strip(),
                'name': match.group(3).strip()
            }

        # Try pattern: name - quantity unit (like "Buttermilk - ½ Gallon Buttermilk")
        match2 = re.match(r'^([A-Za-z\s]+)\s*[-:]\s*([\d\s\/\.\-\u00BC-\u00BE\u2150-\u215E]+)\s+(.+)$', text)
        if match2:
            return {
                'quantity': match2.group(2).strip(),
                'unit': '',
                'name': match2.group(1).strip()
            }

        # Just return as name if can't parse
        return {
            'quantity': '',
            'unit': '',
            'name': text
        }

    def _categorize_recipe(self, name: str) -> str:
        """Categorize recipe by name"""
        name_lower = name.lower()

        if 'soup' in name_lower or 'base' in name_lower or 'jus' in name_lower or 'broth' in name_lower:
            return 'Soup/Base'
        elif 'cheese' in name_lower or 'queso' in name_lower or 'sauce' in name_lower or 'aioli' in name_lower or 'spread' in name_lower:
            return 'Sauce/Cheese'
        elif 'salsa' in name_lower or 'pico' in name_lower or 'verde' in name_lower or 'chimichurri' in name_lower:
            return 'Salsa/Relish'
        elif 'rub' in name_lower or 'flour' in name_lower or 'seasoning' in name_lower:
            return 'Seasoning/Rub'
        elif 'brine' in name_lower or 'marinade' in name_lower:
            return 'Brine/Marinade'
        elif 'batter' in name_lower or 'dough' in name_lower or 'bread' in name_lower:
            return 'Batter/Bread'
        elif 'dressing' in name_lower or 'vinaigrette' in name_lower:
            return 'Dressing'
        elif 'jam' in name_lower or 'butter' in name_lower or 'oil' in name_lower:
            return 'Condiment'
        elif 'pickle' in name_lower or 'slaw' in name_lower:
            return 'Sides'
        else:
            return 'Other'

    def export_to_dict(self) -> Dict:
        """Export recipes in format compatible with recipe manager"""
        return {
            'recipes': [
                {
                    'name': r['name'],
                    'category': r['category'],
                    'base_yield': r.get('yield', 'Not specified'),
                    'scaled_yield': r.get('yield', 'Not specified'),
                    'scale_factor': 1.0,
                    'ingredients': [
                        {
                            'name': ing['name'],
                            'base_qty': ing['quantity'],
                            'unit': ing['unit'],
                            'scaled_qty': ing['quantity']
                        }
                        for ing in r['ingredients']
                    ],
                    'instructions': r.get('instructions', [])
                }
                for r in self.recipes
            ]
        }


if __name__ == "__main__":
    # Test the importer
    importer = DocxRecipeImporter()

    docx_path = "/Users/seanburdges/Desktop/LARIAT/Lariat Recipe Book.docx"
    recipes = importer.import_recipe_book(docx_path)

    print("\n" + "=" * 80)
    print("LARIAT RECIPE BOOK - IMPORT RESULTS")
    print("=" * 80)

    print(f"\nTotal Recipes: {len(recipes)}")

    # Show recipes by category
    by_category = {}
    for recipe in recipes:
        cat = recipe['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(recipe)

    print(f"\nRecipes by Category:")
    for category, recs in sorted(by_category.items()):
        print(f"\n{category}: ({len(recs)} recipes)")
        for recipe in recs:
            print(f"  • {recipe['name']}")
            print(f"    Yield: {recipe.get('yield', 'Not specified')}")
            print(f"    Ingredients: {len(recipe['ingredients'])}")
            print(f"    Instructions: {len(recipe['instructions'])} steps")

    # Show first 3 sample recipes in detail
    if recipes:
        print("\n" + "=" * 80)
        print("SAMPLE RECIPES (First 3)")
        print("=" * 80)

        for sample in recipes[:3]:
            print(f"\n{'=' * 80}")
            print(f"Name: {sample['name']}")
            print(f"Category: {sample['category']}")
            print(f"Yield: {sample.get('yield', 'Not specified')}")
            print(f"\nIngredients ({len(sample['ingredients'])}):")
            for ing in sample['ingredients'][:10]:
                qty_unit = f"{ing['quantity']} {ing['unit']}".strip()
                print(f"  • {qty_unit} {ing['name']}" if qty_unit else f"  • {ing['name']}")
            if len(sample['ingredients']) > 10:
                print(f"  ... and {len(sample['ingredients']) - 10} more")

            if sample['instructions']:
                print(f"\nInstructions ({len(sample['instructions'])} steps):")
                for i, step in enumerate(sample['instructions'][:5], 1):
                    print(f"  {i}. {step[:100]}{'...' if len(step) > 100 else ''}")
                if len(sample['instructions']) > 5:
                    print(f"  ... and {len(sample['instructions']) - 5} more steps")
