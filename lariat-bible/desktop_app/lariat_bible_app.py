#!/usr/bin/env python3
"""
Lariat Bible - Complete Restaurant Management System
Integrates: Recipes, Vendor Management, Order Generation, Price Comparison
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import sys
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from features.recipe_manager import RecipeManager
from features.vendor_manager import VendorManager, PriceComparisonTool
from data_importers.order_guide_parser import OrderGuideParser


class LariatBibleApp:
    """Main Application Window"""

    def __init__(self, root):
        self.root = root
        self.root.title("Lariat Bible - Restaurant Management System")
        self.root.geometry("1400x900")

        # Initialize managers
        self.recipe_manager = None
        self.vendor_manager = None
        # Use official order guide with correct product numbers
        self.order_guide_path = "/Users/seanburdges/Desktop/LARIAT ORDER GUIDE OFFICIAL 8-28-25 (1).xlsx"
        self.use_official_format = True  # Use official combined format

        # Create UI
        self.create_menu()
        self.create_main_interface()
        self.load_data()

    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Order Guide", command=self.load_order_guide)
        file_menu.add_command(label="Export Data", command=self.export_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Generate Price Report", command=self.generate_price_report)
        tools_menu.add_command(label="Run Purchase Analysis", command=self.run_purchase_analysis)
        tools_menu.add_command(label="Refresh Data", command=self.load_data)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Quick Start", command=self.show_quickstart)

    def create_main_interface(self):
        """Create main tabbed interface"""
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Create tabs
        self.create_dashboard_tab()
        self.create_recipes_tab()
        self.create_vendors_tab()
        self.create_orders_tab()
        self.create_analysis_tab()

        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_dashboard_tab(self):
        """Dashboard overview tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📊 Dashboard")

        # Title
        title = tk.Label(tab, text="Lariat Bible Restaurant Management System",
                        font=("Arial", 24, "bold"))
        title.pack(pady=20)

        # Stats frame
        stats_frame = tk.Frame(tab)
        stats_frame.pack(pady=20)

        # Create stat boxes
        self.stat_recipes = self.create_stat_box(stats_frame, "Recipes", "0", 0, 0)
        self.stat_vendors = self.create_stat_box(stats_frame, "Vendor Products", "0", 0, 1)
        self.stat_matches = self.create_stat_box(stats_frame, "Product Matches", "0", 1, 0)
        self.stat_orders = self.create_stat_box(stats_frame, "Orders Generated", "0", 1, 1)

        # Quick actions
        actions_frame = tk.LabelFrame(tab, text="Quick Actions", font=("Arial", 12, "bold"))
        actions_frame.pack(pady=20, padx=20, fill='both', expand=True)

        btn_frame = tk.Frame(actions_frame)
        btn_frame.pack(pady=20)

        tk.Button(btn_frame, text="📖 View Recipes", command=lambda: self.notebook.select(1),
                 font=("Arial", 12), width=20, height=2).grid(row=0, column=0, padx=10, pady=5)
        tk.Button(btn_frame, text="🏪 Manage Vendors", command=lambda: self.notebook.select(2),
                 font=("Arial", 12), width=20, height=2).grid(row=0, column=1, padx=10, pady=5)
        tk.Button(btn_frame, text="📋 Generate Order", command=lambda: self.notebook.select(3),
                 font=("Arial", 12), width=20, height=2).grid(row=1, column=0, padx=10, pady=5)
        tk.Button(btn_frame, text="📈 View Analysis", command=lambda: self.notebook.select(4),
                 font=("Arial", 12), width=20, height=2).grid(row=1, column=1, padx=10, pady=5)

        # System status
        self.status_text = scrolledtext.ScrolledText(tab, height=10, width=80)
        self.status_text.pack(pady=10, padx=20)
        self.log_status("System initialized. Loading data...")

    def create_stat_box(self, parent, label, value, row, col):
        """Create a stat display box"""
        frame = tk.Frame(parent, relief=tk.RAISED, borderwidth=2, bg='#f0f0f0')
        frame.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')

        tk.Label(frame, text=label, font=("Arial", 10), bg='#f0f0f0').pack(pady=5)
        value_label = tk.Label(frame, text=value, font=("Arial", 20, "bold"), bg='#f0f0f0')
        value_label.pack(pady=5)

        # Make it responsive
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        return value_label

    def create_recipes_tab(self):
        """Recipe management tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📖 Recipes")

        # Will be populated by recipe manager
        self.recipe_container = tab

    def create_vendors_tab(self):
        """Vendor management tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🏪 Vendors")

        # Top controls
        control_frame = tk.Frame(tab)
        control_frame.pack(fill='x', padx=10, pady=5)

        tk.Label(control_frame, text="Vendor:", font=("Arial", 10, "bold")).pack(side='left', padx=5)

        self.vendor_filter = ttk.Combobox(control_frame, values=["All", "Sysco", "Shamrock"],
                                         state='readonly', width=15)
        self.vendor_filter.set("All")
        self.vendor_filter.pack(side='left', padx=5)
        self.vendor_filter.bind('<<ComboboxSelected>>', self.filter_vendors)

        tk.Button(control_frame, text="Compare Prices", command=self.compare_prices).pack(side='left', padx=5)
        tk.Button(control_frame, text="Match Products", command=self.match_products).pack(side='left', padx=5)
        tk.Button(control_frame, text="Refresh", command=self.refresh_vendors).pack(side='left', padx=5)

        # Split view: product list and details
        paned = tk.PanedWindow(tab, orient=tk.HORIZONTAL)
        paned.pack(fill='both', expand=True, padx=10, pady=5)

        # Left: Product list
        left_frame = tk.Frame(paned)
        paned.add(left_frame, width=600)

        tk.Label(left_frame, text="Vendor Products", font=("Arial", 12, "bold")).pack()

        # Treeview for products
        tree_frame = tk.Frame(left_frame)
        tree_frame.pack(fill='both', expand=True)

        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side='right', fill='y')

        self.vendor_tree = ttk.Treeview(tree_frame, columns=('Code', 'Description', 'Brand', 'Price', 'Unit'),
                                       yscrollcommand=scrollbar.set)
        self.vendor_tree.pack(fill='both', expand=True)
        scrollbar.config(command=self.vendor_tree.yview)

        self.vendor_tree.heading('#0', text='Vendor')
        self.vendor_tree.heading('Code', text='Code')
        self.vendor_tree.heading('Description', text='Description')
        self.vendor_tree.heading('Brand', text='Brand')
        self.vendor_tree.heading('Price', text='Price')
        self.vendor_tree.heading('Unit', text='Unit')

        self.vendor_tree.column('#0', width=80)
        self.vendor_tree.column('Code', width=100)
        self.vendor_tree.column('Description', width=250)
        self.vendor_tree.column('Brand', width=100)
        self.vendor_tree.column('Price', width=80)
        self.vendor_tree.column('Unit', width=50)

        # Right: Product details
        right_frame = tk.Frame(paned)
        paned.add(right_frame)

        tk.Label(right_frame, text="Product Details", font=("Arial", 12, "bold")).pack()

        self.vendor_details = scrolledtext.ScrolledText(right_frame, height=25, width=50)
        self.vendor_details.pack(fill='both', expand=True, padx=5, pady=5)

    def create_orders_tab(self):
        """Order generation tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📋 Orders")

        # Top section: Recipe selection
        recipe_frame = tk.LabelFrame(tab, text="Select Recipes for Order", font=("Arial", 12, "bold"))
        recipe_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Recipe listbox with scrollbar
        list_frame = tk.Frame(recipe_frame)
        list_frame.pack(fill='both', expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')

        self.order_recipe_list = tk.Listbox(list_frame, selectmode='multiple',
                                           yscrollcommand=scrollbar.set, height=10)
        self.order_recipe_list.pack(fill='both', expand=True)
        scrollbar.config(command=self.order_recipe_list.yview)

        # Controls
        control_frame = tk.Frame(recipe_frame)
        control_frame.pack(fill='x', padx=5, pady=5)

        tk.Label(control_frame, text="Servings Multiplier:", font=("Arial", 10)).pack(side='left', padx=5)
        self.servings_multiplier = tk.Entry(control_frame, width=10)
        self.servings_multiplier.insert(0, "1.0")
        self.servings_multiplier.pack(side='left', padx=5)

        tk.Button(control_frame, text="Generate Order", command=self.generate_order,
                 font=("Arial", 10, "bold"), bg='#4CAF50', fg='white').pack(side='left', padx=5)

        tk.Button(control_frame, text="Clear Selection", command=self.clear_order_selection).pack(side='left', padx=5)

        # Bottom section: Order preview
        order_frame = tk.LabelFrame(tab, text="Generated Order", font=("Arial", 12, "bold"))
        order_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.order_preview = scrolledtext.ScrolledText(order_frame, height=15)
        self.order_preview.pack(fill='both', expand=True, padx=5, pady=5)

        # Order summary
        summary_frame = tk.Frame(order_frame)
        summary_frame.pack(fill='x', padx=5, pady=5)

        self.order_summary_label = tk.Label(summary_frame, text="No order generated yet",
                                           font=("Arial", 10, "italic"))
        self.order_summary_label.pack(side='left')

        tk.Button(summary_frame, text="Export Order", command=self.export_order).pack(side='right', padx=5)

    def create_analysis_tab(self):
        """Purchase analysis tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📈 Analysis")

        # Controls
        control_frame = tk.Frame(tab)
        control_frame.pack(fill='x', padx=10, pady=5)

        tk.Button(control_frame, text="Run Analysis", command=self.run_purchase_analysis,
                 font=("Arial", 10, "bold")).pack(side='left', padx=5)
        tk.Button(control_frame, text="Export Report", command=self.export_analysis).pack(side='left', padx=5)

        # Results display
        results_frame = tk.LabelFrame(tab, text="Analysis Results", font=("Arial", 12, "bold"))
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.analysis_text = scrolledtext.ScrolledText(results_frame, height=30)
        self.analysis_text.pack(fill='both', expand=True, padx=5, pady=5)

    # ========================================================================
    # Data Loading
    # ========================================================================

    def load_data(self):
        """Load all data"""
        try:
            self.log_status("Loading recipes...")
            self.recipe_manager = RecipeManager()

            # Create recipe manager UI in recipes tab
            if self.recipe_manager.recipes:
                self.recipe_manager.create_ui(self.recipe_container)
                self.stat_recipes.config(text=str(len(self.recipe_manager.recipes)))
                self.log_status(f"✓ Loaded {len(self.recipe_manager.recipes)} recipes")

            self.log_status("Loading vendor data...")
            self.vendor_manager = VendorManager(self.order_guide_path, use_official_format=self.use_official_format)

            total_products = len(self.vendor_manager.sysco_products) + len(self.vendor_manager.shamrock_products)
            self.stat_vendors.config(text=str(total_products))
            self.log_status(f"✓ Loaded {total_products} vendor products")

            # Update matched pairs count
            matched_pairs_count = len(self.vendor_manager.official_matches)
            self.stat_matches.config(text=str(matched_pairs_count))
            if matched_pairs_count > 0:
                self.log_status(f"✓ Loaded {matched_pairs_count} pre-matched product pairs (official guide)")

            # Populate vendor tree
            self.populate_vendor_tree()

            # Populate order recipe list
            self.populate_order_recipes()

            self.log_status("✓ All data loaded successfully!")
            status_msg = f"Ready - {len(self.recipe_manager.recipes)} recipes, {total_products} products"
            if matched_pairs_count > 0:
                status_msg += f", {matched_pairs_count} matched pairs"
            self.status_bar.config(text=status_msg)

        except Exception as e:
            self.log_status(f"❌ Error loading data: {e}")
            messagebox.showerror("Error", f"Failed to load data: {e}")

    def load_order_guide(self):
        """Load new order guide file"""
        filename = filedialog.askopenfilename(
            title="Select Order Guide",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )

        if filename:
            try:
                self.order_guide_path = filename
                # Auto-detect if it's official format based on filename
                use_official = "OFFICIAL" in filename.upper()
                self.use_official_format = use_official
                self.vendor_manager = VendorManager(filename, use_official_format=use_official)
                self.populate_vendor_tree()
                format_str = "official format" if use_official else "standard format"
                self.log_status(f"✓ Loaded order guide ({format_str}): {filename}")
                messagebox.showinfo("Success", "Order guide loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load order guide: {e}")

    # ========================================================================
    # Vendor Functions
    # ========================================================================

    def populate_vendor_tree(self):
        """Populate vendor product tree"""
        if not self.vendor_manager:
            return

        # Clear existing
        for item in self.vendor_tree.get_children():
            self.vendor_tree.delete(item)

        # Add Sysco products
        for product in self.vendor_manager.sysco_products[:100]:  # Limit to first 100 for performance
            self.vendor_tree.insert('', 'end', text='Sysco', values=(
                product.get('product_code', ''),
                product.get('description', 'N/A')[:30],
                product.get('brand', '')[:20],
                '',
                product.get('unit', '')
            ))

        # Add Shamrock products
        for product in self.vendor_manager.shamrock_products[:100]:
            price = product.get('price', 0)
            price_str = f"${price:.2f}" if price else ''

            self.vendor_tree.insert('', 'end', text='Shamrock', values=(
                product.get('product_code', ''),
                product.get('description', '')[:30],
                product.get('brand', '')[:20],
                price_str,
                product.get('unit', '')
            ))

    def filter_vendors(self, event=None):
        """Filter vendors based on selection"""
        # Re-populate based on filter
        self.populate_vendor_tree()

    def refresh_vendors(self):
        """Refresh vendor data"""
        self.populate_vendor_tree()
        self.log_status("✓ Vendor data refreshed")

    def compare_prices(self):
        """Run price comparison"""
        if not self.vendor_manager:
            messagebox.showwarning("Warning", "Please load vendor data first")
            return

        try:
            comparisons = self.vendor_manager.compare_prices()
            self.stat_matches.config(text=str(len(comparisons)))

            result = f"Price Comparison Results\n{'=' * 50}\n\n"
            result += f"Total Comparisons: {len(comparisons)}\n\n"

            for i, comp in enumerate(comparisons[:20], 1):
                result += f"{i}. {comp['description']}\n"
                result += f"   Sysco: ${comp['sysco_price']:.2f} | Shamrock: ${comp['shamrock_price']:.2f}\n"
                result += f"   Difference: ${comp['difference']:.2f} ({comp['cheaper_vendor']} is cheaper)\n\n"

            self.vendor_details.delete(1.0, tk.END)
            self.vendor_details.insert(1.0, result)

            self.log_status(f"✓ Found {len(comparisons)} price comparisons")

        except Exception as e:
            messagebox.showerror("Error", f"Price comparison failed: {e}")

    def match_products(self):
        """Match products across vendors"""
        if not self.vendor_manager:
            messagebox.showwarning("Warning", "Please load vendor data first")
            return

        try:
            matches = self.vendor_manager.match_products(min_confidence=0.5)
            self.stat_matches.config(text=str(len(matches)))

            result = f"Product Matching Results\n{'=' * 50}\n\n"
            result += f"Total Matches: {len(matches)}\n\n"

            for i, match in enumerate(matches[:20], 1):
                result += f"{i}. {match['description']} (Confidence: {match['confidence']})\n"
                result += f"   Shamrock Code: {match['shamrock_product']['product_code']}\n"
                if match['sysco_product']:
                    result += f"   Sysco Code: {match['sysco_product']['product_code']}\n"
                result += "\n"

            self.vendor_details.delete(1.0, tk.END)
            self.vendor_details.insert(1.0, result)

            self.log_status(f"✓ Found {len(matches)} product matches")

        except Exception as e:
            messagebox.showerror("Error", f"Product matching failed: {e}")

    # ========================================================================
    # Order Functions
    # ========================================================================

    def populate_order_recipes(self):
        """Populate recipe list for ordering"""
        if not self.recipe_manager or not self.recipe_manager.recipes:
            return

        self.order_recipe_list.delete(0, tk.END)
        for recipe in self.recipe_manager.recipes:
            self.order_recipe_list.insert(tk.END, recipe['name'])

    def generate_order(self):
        """Generate order from selected recipes"""
        if not self.vendor_manager or not self.recipe_manager:
            messagebox.showwarning("Warning", "Please load data first")
            return

        # Get selected recipes
        selection = self.order_recipe_list.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select at least one recipe")
            return

        selected_recipes = [self.recipe_manager.recipes[i] for i in selection]

        try:
            multiplier = float(self.servings_multiplier.get())
        except:
            multiplier = 1.0

        try:
            order = self.vendor_manager.generate_order(selected_recipes, multiplier)

            # Display order
            result = f"GENERATED ORDER\n{'=' * 60}\n\n"
            result += f"Recipes: {order['recipe_count']}\n"
            result += f"Servings Multiplier: {multiplier}x\n"
            result += f"Total Items: {order['total_items']}\n\n"

            result += f"SYSCO ORDER ({len(order['sysco_order'])} items)\n{'-' * 60}\n"
            for code, item in list(order['sysco_order'].items())[:10]:
                result += f"{item['product'].get('brand')} - Qty: {item['quantity']:.1f}\n"

            result += f"\nSHAMROCK ORDER ({len(order['shamrock_order'])} items)\n{'-' * 60}\n"
            for code, item in list(order['shamrock_order'].items())[:10]:
                result += f"{item['product'].get('description')}\n"
                result += f"  Qty: {item['quantity']:.1f} @ ${item['product'].get('price', 0):.2f}\n"

            result += f"\nESTIMATED TOTAL: ${order['estimated_sysco_cost'] + order['estimated_shamrock_cost']:.2f}\n"

            self.order_preview.delete(1.0, tk.END)
            self.order_preview.insert(1.0, result)

            self.order_summary_label.config(
                text=f"Order: {order['total_items']} items - ${order['estimated_sysco_cost'] + order['estimated_shamrock_cost']:.2f}"
            )

            self.stat_orders.config(text=str(int(self.stat_orders.cget('text')) + 1))
            self.log_status(f"✓ Generated order: {order['total_items']} items")

            # Store for export
            self.current_order = order

        except Exception as e:
            messagebox.showerror("Error", f"Order generation failed: {e}")
            import traceback
            traceback.print_exc()

    def clear_order_selection(self):
        """Clear order selection"""
        self.order_recipe_list.selection_clear(0, tk.END)

    def export_order(self):
        """Export current order"""
        if not hasattr(self, 'current_order'):
            messagebox.showwarning("Warning", "No order to export")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.current_order, f, indent=2, default=str)
                messagebox.showinfo("Success", f"Order exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")

    # ========================================================================
    # Analysis Functions
    # ========================================================================

    def run_purchase_analysis(self):
        """Run purchase history analysis"""
        if not self.vendor_manager:
            messagebox.showwarning("Warning", "Please load vendor data first")
            return

        try:
            analysis = self.vendor_manager.analyze_purchase_history()

            result = f"PURCHASE HISTORY ANALYSIS\n{'=' * 60}\n\n"
            result += f"Products with History: {analysis['total_products_with_history']}\n\n"

            result += f"TOP 20 MOST PURCHASED PRODUCTS\n{'-' * 60}\n"
            for i, item in enumerate(analysis['top_purchased_products'][:20], 1):
                product = item['product']
                result += f"{i}. {product.get('brand')} - {product.get('pack')} {product.get('size')}\n"
                result += f"   Total Purchases: {item['total_purchases']:.1f} units\n\n"

            result += f"\nRECOMMENDATIONS\n{'-' * 60}\n"
            for rec in analysis['recommendations']:
                result += f"• {rec}\n"

            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(1.0, result)

            self.log_status("✓ Purchase analysis complete")

        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()

    def export_analysis(self):
        """Export analysis results"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            try:
                content = self.analysis_text.get(1.0, tk.END)
                with open(filename, 'w') as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Analysis exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")

    # ========================================================================
    # Utility Functions
    # ========================================================================

    def log_status(self, message):
        """Log status message"""
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.root.update()

    def generate_price_report(self):
        """Generate price comparison report"""
        if not self.vendor_manager:
            messagebox.showwarning("Warning", "Please load vendor data first")
            return

        try:
            tool = PriceComparisonTool(self.vendor_manager)
            tool.generate_report('price_comparison_report.txt')
            messagebox.showinfo("Success", "Price report generated: price_comparison_report.txt")
            self.log_status("✓ Generated price comparison report")
        except Exception as e:
            messagebox.showerror("Error", f"Report generation failed: {e}")

    def export_data(self):
        """Export all data"""
        messagebox.showinfo("Export", "Export functionality - to be implemented")

    def show_about(self):
        """Show about dialog"""
        about_text = """
Lariat Bible - Restaurant Management System
Version 1.0

Features:
• Recipe Management (53 recipes)
• Vendor Management (659 products)
• Price Comparison
• Product Matching
• Order Generation
• Purchase Analysis

Created: November 2025
        """
        messagebox.showinfo("About Lariat Bible", about_text)

    def show_quickstart(self):
        """Show quick start guide"""
        quickstart = """
Quick Start Guide:

1. RECIPES TAB
   - Browse 53 recipes from Lariat Recipe Book
   - Search by name or category
   - View ingredients and instructions

2. VENDORS TAB
   - View Sysco and Shamrock products
   - Compare prices between vendors
   - Match products across vendors

3. ORDERS TAB
   - Select recipes to order
   - Set servings multiplier
   - Generate consolidated vendor orders
   - Export orders to JSON

4. ANALYSIS TAB
   - Analyze purchase history
   - View top purchased products
   - Get recommendations

For more help, see VENDOR_MANAGEMENT_COMPLETE.md
        """
        messagebox.showinfo("Quick Start", quickstart)


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = LariatBibleApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
