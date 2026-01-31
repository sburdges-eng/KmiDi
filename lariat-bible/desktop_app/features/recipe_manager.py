"""
Recipe Manager Module
Integrates with enhanced data importers to display recipes and ingredients
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_importers import ExcelImporter
from data_importers.docx_importer import DocxRecipeImporter
import os


class RecipeManager:
    def __init__(self):
        self.excel_importer = ExcelImporter()
        self.docx_importer = DocxRecipeImporter()
        self.recipes = None
        self.ingredients = None
        self.load_data()

    def load_data(self):
        """Load recipes from Word document and ingredients from Excel"""
        try:
            # Load recipes from Word document (PRIMARY SOURCE)
            docx_path = "/Users/seanburdges/Desktop/LARIAT/Lariat Recipe Book.docx"
            if os.path.exists(docx_path):
                recipe_list = self.docx_importer.import_recipe_book(docx_path)
                recipes_data = self.docx_importer.export_to_dict()
                self.recipes = recipes_data['recipes']
                print(f"✓ Loaded {len(self.recipes)} recipes from Word document")
            else:
                print(f"⚠ Word document not found at {docx_path}")
                # Fallback to Excel if Word doc not available
                recipes_data = self.excel_importer.import_menu_bible_recipes()
                if recipes_data:
                    self.recipes = recipes_data['recipes']
                    print(f"✓ Loaded {len(self.recipes)} recipes from Excel (fallback)")
                else:
                    self.recipes = []
                    print("⚠ No recipes loaded")

            # Load ingredient database from Excel
            self.ingredients = self.excel_importer.import_ingredient_database()
            if self.ingredients:
                print(f"✓ Loaded {len(self.ingredients)} ingredients")
            else:
                self.ingredients = []
                print("⚠ No ingredients loaded")

        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            self.recipes = []
            self.ingredients = []

    def get_recipes_by_category(self):
        """Get recipes organized by category"""
        by_category = {}
        for recipe in self.recipes:
            category = recipe.get('category', 'Unknown')
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(recipe)
        return by_category

    def get_recipe_by_name(self, name):
        """Get a specific recipe by name"""
        for recipe in self.recipes:
            if recipe['name'].lower() == name.lower():
                return recipe
        return None

    def search_recipes(self, query):
        """Search recipes by name"""
        query = query.lower()
        return [r for r in self.recipes if query in r['name'].lower()]

    def get_ingredient_by_name(self, name):
        """Get ingredient pricing info"""
        for ing in self.ingredients:
            if ing['name'].lower() == name.lower():
                return ing
        return None

    def create_ui(self, parent_frame):
        """Create the recipe manager UI"""
        # Main container with padding
        main_container = ttk.Frame(parent_frame)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)

        # Title
        title = ttk.Label(
            main_container,
            text="Recipe Manager",
            font=('Arial', 18, 'bold')
        )
        title.pack(pady=(0, 10))

        # Create two-column layout
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill='both', expand=True)

        # Left panel - Recipe list
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))

        # Search bar
        search_frame = ttk.Frame(left_panel)
        search_frame.pack(fill='x', pady=(0, 10))

        ttk.Label(search_frame, text="Search:").pack(side='left', padx=(0, 5))
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        search_entry.bind('<KeyRelease>', lambda e: self.filter_recipes())

        # Category filter
        filter_frame = ttk.Frame(left_panel)
        filter_frame.pack(fill='x', pady=(0, 10))

        ttk.Label(filter_frame, text="Category:").pack(side='left', padx=(0, 5))

        categories = ['All'] + sorted(list(set(r.get('category', 'Unknown') for r in self.recipes)))
        self.category_var = tk.StringVar(value='All')
        category_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.category_var,
            values=categories,
            state='readonly',
            width=20
        )
        category_combo.pack(side='left')
        category_combo.bind('<<ComboboxSelected>>', lambda e: self.filter_recipes())

        # Recipe listbox
        list_frame = ttk.LabelFrame(left_panel, text="Recipes", padding=5)
        list_frame.pack(fill='both', expand=True)

        # Scrollbar for recipe list
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')

        self.recipe_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=('Arial', 10),
            selectmode='single'
        )
        self.recipe_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.recipe_listbox.yview)

        self.recipe_listbox.bind('<<ListboxSelect>>', self.on_recipe_select)

        # Right panel - Recipe details
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side='right', fill='both', expand=True, padx=(5, 0))

        details_frame = ttk.LabelFrame(right_panel, text="Recipe Details", padding=10)
        details_frame.pack(fill='both', expand=True)

        # Recipe details text area
        self.details_text = scrolledtext.ScrolledText(
            details_frame,
            wrap=tk.WORD,
            font=('Arial', 10),
            padx=10,
            pady=10
        )
        self.details_text.pack(fill='both', expand=True)

        # Populate initial list
        self.populate_recipe_list()

    def populate_recipe_list(self):
        """Populate the recipe listbox"""
        self.recipe_listbox.delete(0, tk.END)

        for recipe in self.recipes:
            display_name = f"{recipe['name']} ({recipe.get('category', 'Unknown')})"
            self.recipe_listbox.insert(tk.END, display_name)

    def filter_recipes(self):
        """Filter recipes based on search and category"""
        search_query = self.search_var.get().lower()
        category = self.category_var.get()

        self.recipe_listbox.delete(0, tk.END)

        for recipe in self.recipes:
            # Filter by category
            if category != 'All' and recipe.get('category', 'Unknown') != category:
                continue

            # Filter by search query
            if search_query and search_query not in recipe['name'].lower():
                continue

            display_name = f"{recipe['name']} ({recipe.get('category', 'Unknown')})"
            self.recipe_listbox.insert(tk.END, display_name)

    def on_recipe_select(self, event):
        """Handle recipe selection"""
        selection = self.recipe_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        recipe_name = self.recipe_listbox.get(index).split(' (')[0]

        recipe = self.get_recipe_by_name(recipe_name)
        if recipe:
            self.display_recipe_details(recipe)

    def display_recipe_details(self, recipe):
        """Display detailed recipe information"""
        self.details_text.delete('1.0', tk.END)

        # Recipe header
        self.details_text.insert(tk.END, f"{recipe['name']}\n", 'title')
        self.details_text.insert(tk.END, "=" * 60 + "\n\n")

        # Category and yield
        self.details_text.insert(tk.END, f"Category: ", 'bold')
        self.details_text.insert(tk.END, f"{recipe.get('category', 'Unknown')}\n")

        self.details_text.insert(tk.END, f"Yield: ", 'bold')
        self.details_text.insert(tk.END, f"{recipe.get('base_yield', 'N/A')}\n\n")

        # Ingredients header
        self.details_text.insert(tk.END, "INGREDIENTS\n", 'header')
        self.details_text.insert(tk.END, "-" * 60 + "\n\n")

        # List ingredients
        total_cost = 0
        for i, ing in enumerate(recipe['ingredients'], 1):
            qty = ing.get('scaled_qty', 0)
            unit = ing.get('unit', '')
            name = ing.get('name', '')

            self.details_text.insert(tk.END, f"{i}. ")
            self.details_text.insert(tk.END, f"{name}\n", 'ingredient')
            self.details_text.insert(tk.END, f"   Amount: {qty} {unit}\n")

            # Try to get pricing info
            pricing = self.get_ingredient_by_name(name)
            if pricing and pricing.get('cost_per_package'):
                cost_pkg = pricing.get('cost_per_package', 0)
                category = pricing.get('category', 'N/A')
                self.details_text.insert(tk.END, f"   Cost: ${cost_pkg:.2f} ({category})\n")
                total_cost += cost_pkg

            self.details_text.insert(tk.END, "\n")

        # Total estimated cost
        if total_cost > 0:
            self.details_text.insert(tk.END, "\n" + "=" * 60 + "\n")
            self.details_text.insert(tk.END, f"Estimated Total Cost: ${total_cost:.2f}\n", 'cost')

        # Configure text tags for styling
        self.details_text.tag_config('title', font=('Arial', 14, 'bold'))
        self.details_text.tag_config('header', font=('Arial', 12, 'bold'))
        self.details_text.tag_config('bold', font=('Arial', 10, 'bold'))
        self.details_text.tag_config('ingredient', font=('Arial', 10, 'bold'), foreground='#2C3E50')
        self.details_text.tag_config('cost', font=('Arial', 11, 'bold'), foreground='#27AE60')
