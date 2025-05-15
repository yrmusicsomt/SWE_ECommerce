# SWE_ECommerce
# Online Electronics Store - Python/FastAPI Implementation Guide (Assignment 3)

## 1. Introduction

This document provides a detailed guide for implementing the Online Electronics Store system for SWE30003 Assignment 3 using Python and the FastAPI framework. It builds upon the Object-Oriented Design from Assignment 2 and incorporates Stripe API for payments, Austpost API for shipping, and MongoDB for the database.

The primary design patterns to be explicitly implemented are:
1.  **Factory Pattern**: For the creation of `User`-related objects (`Customer`, `Admin`).
2.  **Strategy Pattern**: For handling different payment methods, specifically `StripePaymentStrategy`.

## 2. Core Design Principles

* **Encapsulated Service Interactions**: Dedicated service classes (`StripeService`, `AustpostService`, `MongoDBService`) will manage all direct communications with external APIs and the database.
* **Domain-Driven Focus**: Core domain classes (e.g., `Order`, `Product`, `Customer`) remain central.
* **API-First with FastAPI**: Leverage FastAPI for building robust and well-documented APIs, using Pydantic models for data validation and serialization.
* **Dependency Injection**: Utilize FastAPI's dependency injection for managing service instances.

## 3. Project Structure (Suggested)

online-electronics-store/├── app/│   ├── init.py│   ├── main.py                 # FastAPI app instance, bootstrap logic, startup events│   ├── core/│   │   ├── init.py│   │   └── config.py           # Application settings (e.g., DB URI, API keys from .env)│   ├── models/                 # Pydantic models for API requests/responses & domain data structures│   │   ├── init.py│   │   ├── product_models.py   # Product schemas│   │   ├── order_models.py     # Order, CartItem, Shipment, Address schemas│   │   ├── user_models.py      # User, Customer, Admin schemas│   │   └── payment_models.py   # Payment, PurchaseRecord schemas│   ├── domain_logic/           # Core business logic classes (can be merged with models if simple)│   │   ├── init.py│   │   ├── product.py│   │   ├── order.py│   │   ├── user.py             # Person/User base, Customer, Admin classes│   │   ├── shopping_cart.py│   │   └── payment_context.py  # Payment context class for Strategy│   ├── services/               # External API and DB interaction│   │   ├── init.py│   │   ├── stripe_service.py│   │   ├── austpost_service.py│   │   └── mongodb_service.py│   ├── factories/              # Factory Pattern implementations│   │   ├── init.py│   │   └── user_factory.py│   ├── strategies/             # Strategy Pattern implementations│   │   ├── init.py│   │   └── payment_strategy.py # PaymentMethod interface (ABC) and StripePaymentStrategy│   ├── routers/                # FastAPI routers (API endpoints)│   │   ├── init.py│   │   ├── products_router.py│   │   ├── cart_router.py│   │   ├── orders_router.py│   │   └── users_router.py│   └── dependencies.py         # Common FastAPI dependencies (e.g., get_db_session)├── tests/                    # Unit and integration tests│   ├── init.py│   └── ...├── .env                      # Environment variables (DB_CONNECTION_STRING, etc.)├── requirements.txt          # Python dependencies (fastapi, uvicorn, pydantic, pymongo, python-dotenv)└── README.md
## 4. Key Classes and Responsibilities

(Python uses Pydantic for data classes/models and Abstract Base Classes (ABCs) for interfaces.)

### 4.1. Models (`app/models/`) - Pydantic Schemas

These define the structure of data for API requests, responses, and internal use.

* `user_models.py`:
    * `UserBase(BaseModel)`: Common user attributes (id, email, name).
    * `CustomerCreate(UserBase)`: For creating customers.
    * `Customer(UserBase)`: Represents a customer, potentially with `customer_id`.
    * `AdminCreate(UserBase)`: For creating admins.
    * `Admin(UserBase)`: Represents an admin, potentially with `admin_id`.
* `product_models.py`:
    * `ProductBase(BaseModel)`: name, description, price, category.
    * `ProductCreate(ProductBase)`
    * `Product(ProductBase)`: Includes `id`, `stock_quantity`.
* `order_models.py`:
    * `Address(BaseModel)`: street, city, postcode, state, country.
    * `CartItemBase(BaseModel)`: product_id, quantity.
    * `CartItem(CartItemBase)`: Includes calculated subtotal.
    * `ShoppingCart(BaseModel)`: items: `List[CartItem]`, total: `float`.
    * `OrderBase(BaseModel)`: customer_id, shipping_address: `Address`.
    * `OrderCreate(OrderBase)`: items: `List[CartItemBase]`.
    * `Order(OrderBase)`: Includes `id`, items: `List[CartItem]`, cart_total: `float`, shipping_fee: `float`, grand_total: `float`, status: `str`, created_at: `datetime`.
    * `ShipmentInfo(BaseModel)`: tracking_id: `Optional[str]`, status: `str`.
* `payment_models.py`:
    * `PaymentDetails(BaseModel)`: E.g., card_token (mocked), payment_method_type.
    * `PaymentResponse(BaseModel)`: success: `bool`, transaction_id: `Optional[str]`, message: `str`.
    * `PurchaseRecord(BaseModel)`: id: `str`, order_id: `str`, customer_id: `str`, amount: `float`, payment_method: `str`, transaction_date: `datetime`.

### 4.2. Domain Logic Classes (`app/domain_logic/`)

These classes encapsulate business rules and operations. They might use Pydantic models for their data.

* `user.py`:
    * `User` (could be an abstract base or just a concept): Base attributes.
    * `Customer(User)`: Customer-specific logic.
    * `Admin(User)`: Admin-specific logic.
* `shopping_cart.py`:
    * `ShoppingCartLogic`: Methods to add/remove items, calculate totals. Manages a `models.order_models.ShoppingCart` Pydantic model.
* `order.py`:
    * `OrderLogic`: Manages the lifecycle of an order, calculations, status updates. Manages an `models.order_models.Order` Pydantic model.
* `payment_context.py`:
    * `PaymentContext`: Holds a reference to a `PaymentMethod` strategy. `execute_payment(amount, details)` method.

### 4.3. Factory Pattern Classes (`app/factories/`)

* `user_factory.py`:
    * `UserFactory`:
        ```python
        from app.models.user_models import Customer, Admin, UserType # UserType is an Enum
        from app.domain_logic.user import Customer as CustomerLogic, Admin as AdminLogic

        class UserFactory:
            @staticmethod
            def create_user(user_type: UserType, name: str, email: str, **kwargs):
                if user_type == UserType.CUSTOMER:
                    # In a real app, you might save to DB here or in a service
                    # For now, just creating the domain object
                    print(f"Factory creating Customer: {name}")
                    return CustomerLogic(name=name, email=email, **kwargs) # Or a Pydantic model
                elif user_type == UserType.ADMIN:
                    print(f"Factory creating Admin: {name}")
                    return AdminLogic(name=name, email=email, **kwargs) # Or a Pydantic model
                else:
                    raise ValueError("Invalid user type")
        ```

### 4.4. Strategy Pattern Classes (`app/strategies/`)

* `payment_strategy.py`:
    * `PaymentMethod(ABC)` (Abstract Base Class):
        ```python
        from abc import ABC, abstractmethod
        from app.models.payment_models import PaymentDetails, PaymentResponse

        class PaymentMethod(ABC):
            @abstractmethod
            def process_payment(self, amount: float, details: PaymentDetails) -> PaymentResponse:
                pass
        ```
    * `StripePaymentStrategy(PaymentMethod)`:
        ```python
        from app.services.stripe_service import StripeService # Injected or instantiated

        class StripePaymentStrategy(PaymentMethod):
            def __init__(self, stripe_service: StripeService):
                self.stripe_service = stripe_service

            def process_payment(self, amount: float, details: PaymentDetails) -> PaymentResponse:
                print(f"StripePaymentStrategy: Processing payment of {amount} via Stripe.")
                # Assuming details.card_token is available for Stripe
                return self.stripe_service.process_charge(
                    amount=amount,
                    currency="AUD", # Example currency
                    payment_token=details.card_token # Mocked token
                )
        ```

### 4.5. Service Interaction Classes (`app/services/`)

* `stripe_service.py` (Mock):
    ```python
    from app.models.payment_models import PaymentResponse

    class StripeService:
        def __init__(self):
            print("MockStripeService Initialized.")

        def process_charge(self, amount: float, currency: str, payment_token: str) -> PaymentResponse:
            print(f"[MockStripeService] Attempting to charge {amount:.2f} {currency} with token {payment_token}")
            # Simulate processing
            if amount > 0 and currency and payment_token:
                print(f"[MockStripeService] Charge successful for {amount:.2f} {currency}")
                return PaymentResponse(success=True, transaction_id="mock_stripe_txn_123", message="Payment successful (mocked)")
            else:
                print("[MockStripeService] Charge failed due to invalid parameters.")
                return PaymentResponse(success=False, message="Payment failed: Invalid parameters (mocked)")
    ```
* `austpost_service.py` (Mock):
    ```python
    from app.models.order_models import Address # Assuming Address is a Pydantic model

    class AustpostService:
        def __init__(self):
            print("MockAustpostService Initialized.")

        def get_shipping_quote(self, destination_address: Address, package_weight_kg: float) -> float:
            print(f"[MockAustpostService] Calculating shipping for address: {destination_address.postcode}, weight: {package_weight_kg:.2f} kg")
            fee = 10.0  # Base fee
            if destination_address.postcode.startswith("3"): fee += 5.0
            elif destination_address.postcode.startswith("2"): fee += 7.0
            else: fee += 10.0
            if package_weight_kg > 5: fee += (package_weight_kg - 5) * 0.5
            print(f"[MockAustpostService] Calculated shipping fee: {fee:.2f} AUD")
            return round(fee, 2)
    ```
* `mongodb_service.py`:
    ```python
    from pymongo import MongoClient
    from pymongo.database import Database
    from app.core.config import settings # Assuming settings.MONGODB_URL and settings.MONGODB_DB_NAME
    from app.models.order_models import Order # Pydantic model
    from app.models.payment_models import PurchaseRecord # Pydantic model
    from app.models.product_models import Product # Pydantic model
    from typing import List, Optional
    import uuid # For generating IDs

    class MongoDBService:
        client: MongoClient = None
        db: Database = None

        def connect_to_database(self):
            print("Connecting to MongoDB...")
            self.client = MongoClient(settings.MONGODB_URL)
            self.db = self.client[settings.MONGODB_DB_NAME]
            print(f"Successfully connected to MongoDB: {settings.MONGODB_DB_NAME}")

        def close_database_connection(self):
            if self.client:
                print("Closing MongoDB connection...")
                self.client.close()

        def get_db(self) -> Database: # For direct access if needed, or for dependencies
            if not self.db:
                self.connect_to_database() # Should ideally be handled by startup event
            return self.db

        # --- Product Operations ---
        def get_product_by_id(self, product_id: str) -> Optional[Product]:
            product_data = self.db.products.find_one({"id": product_id})
            return Product(**product_data) if product_data else None

        def get_all_products(self) -> List[Product]:
            products_data = list(self.db.products.find())
            return [Product(**p) for p in products_data]

        def create_product(self, product_create: ProductCreate) -> Product: # ProductCreate is Pydantic model
            product_dict = product_create.model_dump()
            product_dict["id"] = str(uuid.uuid4()) # Generate unique ID
            product_dict["stock_quantity"] = product_dict.get("stock_quantity", 0) # Default stock
            self.db.products.insert_one(product_dict)
            return Product(**product_dict)

        def update_product_stock(self, product_id: str, quantity_change: int) -> bool:
            result = self.db.products.update_one(
                {"id": product_id},
                {"$inc": {"stock_quantity": quantity_change}}
            )
            return result.modified_count > 0

        # --- Order Operations ---
        def save_order(self, order: Order) -> Order: # Order is Pydantic model
            order_dict = order.model_dump(by_alias=True) # Use by_alias if you have field aliases
            if not order_dict.get("id"):
                 order_dict["id"] = str(uuid.uuid4())
            self.db.orders.update_one(
                {"id": order_dict["id"]},
                {"$set": order_dict},
                upsert=True
            )
            return Order(**order_dict) # Return the saved/updated order

        def get_order_by_id(self, order_id: str) -> Optional[Order]:
            order_data = self.db.orders.find_one({"id": order_id})
            return Order(**order_data) if order_data else None

        # --- Purchase Record Operations ---
        def save_purchase_record(self, record: PurchaseRecord) -> PurchaseRecord:
            record_dict = record.model_dump(by_alias=True)
            if not record_dict.get("id"):
                record_dict["id"] = str(uuid.uuid4())
            self.db.purchase_records.insert_one(record_dict)
            return PurchaseRecord(**record_dict)

    # Global instance or managed by FastAPI app state/dependency injection
    mongodb_service = MongoDBService()
    ```

### 4.6. FastAPI Routers (`app/routers/`)

Example: `orders_router.py`
```python
from fastapi import APIRouter, Depends, HTTPException, Body
from typing import List
from app.models.order_models import Order, OrderCreate, CartItemBase, Address
from app.models.payment_models import PaymentDetails, PaymentResponse
from app.services.mongodb_service import mongodb_service, MongoDBService
from app.services.austpost_service import AustpostService
from app.services.stripe_service import StripeService
from app.strategies.payment_strategy import StripePaymentStrategy
from app.domain_logic.payment_context import PaymentContext
from app.domain_logic.shopping_cart import ShoppingCartLogic # For cart total
from app.domain_logic.order import OrderLogic # For order creation and processing
import datetime

router = APIRouter(
    prefix="/orders",
    tags=["orders"],
)

# Dependency for services (can be refined in app.dependencies)
def get_mongodb_service():
    return mongodb_service # Uses the global instance

def get_austpost_service():
    return AustpostService() # New instance or managed singleton

def get_stripe_service():
    return StripeService() # New instance or managed singleton


@router.post("/", response_model=Order)
async def create_order_endpoint(
    order_data: OrderCreate, # Pydantic model for request body
    db: MongoDBService = Depends(get_mongodb_service),
    austpost_svc: AustpostService = Depends(get_austpost_service)
):
    # 1. & 2. Add Product to Cart & Collect Customer Info (Assumed done, customer_id and items provided in order_data)

    # --- Calculate Cart Total ---
    cart_total = 0
    processed_items = []
    for item_base in order_data.items:
        product = await db.get_product_by_id(item_base.product_id) # Make get_product_by_id async if needed
        if not product or product.stock_quantity < item_base.quantity:
            raise HTTPException(status_code=400, detail=f"Product {item_base.product_id} not available or insufficient stock.")
        subtotal = product.price * item_base.quantity
        cart_total += subtotal
        # Create a CartItem with full product details for the order
        processed_items.append({
            "product_id": product.id,
            "product_name": product.name, # Add name to CartItem model if needed
            "quantity": item_base.quantity,
            "unit_price": product.price,
            "subtotal": subtotal
        })

    # 3. Calculate Shipping Fee
    # Assuming package_weight_kg is derived or fixed for mock
    shipping_fee = austpost_svc.get_shipping_quote(order_data.shipping_address, package_weight_kg=2.5)
    grand_total = cart_total + shipping_fee

    # Create the Order domain object/Pydantic model
    new_order_data = {
        "customer_id": order_data.customer_id,
        "items": processed_items, # Use items with full details
        "shipping_address": order_data.shipping_address,
        "cart_total": cart_total,
        "shipping_fee": shipping_fee,
        "grand_total": grand_total,
        "status": "PendingPayment",
        "created_at": datetime.datetime.now(datetime.timezone.utc)
    }
    order_to_save = Order(**new_order_data)
    saved_order = await db.save_order(order_to_save) # Make save_order async if needed
    return saved_order


@router.post("/{order_id}/pay", response_model=PaymentResponse)
async def pay_for_order_endpoint(
    order_id: str,
    payment_details: PaymentDetails, # e.g., {"card_token": "mock_tok_visa", "payment_method_type": "stripe"}
    db: MongoDBService = Depends(get_mongodb_service),
    stripe_svc: StripeService = Depends(get_stripe_service)
):
    order = await db.get_order_by_id(order_id) # Make get_order_by_id async if needed
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    if order.status == "Paid":
        raise HTTPException(status_code=400, detail="Order already paid")

    # 4. Add Payment, Confirm Payment
    if payment_details.payment_method_type.lower() == "stripe":
        stripe_strategy = StripePaymentStrategy(stripe_service=stripe_svc)
        payment_context = PaymentContext(strategy=stripe_strategy)
        payment_result = payment_context.execute_payment(amount=order.grand_total, details=payment_details)
    else:
        raise HTTPException(status_code=400, detail="Unsupported payment method type")

    if payment_result.success:
        order.status = "Paid"
        await db.save_order(order) # Persist updated order status

        # Add Purchase Record
        purchase_record_data = {
            "order_id": order.id,
            "customer_id": order.customer_id,
            "amount": order.grand_total,
            "payment_method": payment_details.payment_method_type,
            "transaction_id": payment_result.transaction_id,
            "transaction_date": datetime.datetime.now(datetime.timezone.utc)
        }
        purchase_rec = PurchaseRecord(**purchase_record_data)
        await db.save_purchase_record(purchase_rec) # Make save_purchase_record async if needed

        # Update stock (simplified)
        for item in order.items:
            await db.update_product_stock(item.product_id, -item.quantity) # Make update_product_stock async

        return payment_result
    else:
        order.status = "PaymentFailed"
        await db.save_order(order)
        raise HTTPException(status_code=400, detail=payment_result.message)

# Add other routers for products, cart, users similarly
4.7. Core Configuration (app/core/config.py)from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv() # Load .env file

class Settings(BaseSettings):
    APP_NAME: str = "AWE Online Electronics Store"
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "awe_store_db")
    # Add other settings as needed

settings = Settings()
Create a .env file in the project root:MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=awe_assignment3_db
5. Detailed Class Interactions & Collaborations (FastAPI Context)Interactions will primarily happen within FastAPI path operation functions (in routers). These functions will:Receive HTTP requests with data validated by Pydantic models.Use injected service instances (MongoDBService, mock API services).Instantiate domain logic classes or use factory/strategy patterns as needed.Call methods on services and domain objects to perform business logic.Return responses, serialized by Pydantic models.Example Flow (Create Order & Pay - simplified from orders_router.py):Client POSTs to /orders/ (Create Order Endpoint):orders_router.create_order_endpoint receives OrderCreate data.It uses MongoDBService to fetch product details for validation and pricing.Calculates cart_total.Calls AustpostService.get_shipping_quote() to get shipping_fee.Calculates grand_total.Creates an Order Pydantic model instance with status "PendingPayment".Calls MongoDBService.save_order() to persist the new order.Returns the created Order details.Client POSTs to /orders/{order_id}/pay (Pay for Order Endpoint):orders_router.pay_for_order_endpoint receives PaymentDetails.Uses MongoDBService to fetch the Order.Strategy Pattern:Instantiates StripePaymentStrategy (passing the StripeService instance).Creates PaymentContext with this strategy.Calls payment_context.execute_payment().StripePaymentStrategy.process_payment() is called.This calls StripeService.process_charge() (mocked).If payment is successful (mock StripeService returns success):Updates Order status to "Paid" via MongoDBService.Creates a PurchaseRecord and saves it via MongoDBService.Updates product stock via MongoDBService.Returns a success PaymentResponse.If payment fails, updates order status and returns an error.Factory Pattern Usage:While not directly in the order flow above, if you have an endpoint to create users (e.g., /users/):# In users_router.py
from app.factories.user_factory import UserFactory
from app.models.user_models import UserCreateRequest, UserResponse, UserType # UserCreateRequest includes type

@router.post("/", response_model=UserResponse)
async def create_new_user(
    user_data: UserCreateRequest,
    db: MongoDBService = Depends(get_mongodb_service)
):
    # user_type can come from user_data.user_type
    domain_user = UserFactory.create_user(
        user_type=user_data.type, # e.g., UserType.CUSTOMER
        name=user_data.name,
        email=user_data.email
        # ... other fields
    )
    # Now save the domain_user (or its Pydantic representation) to the database via MongoDBService
    # e.g., saved_db_user = await db.save_user(domain_user.to_pydantic_model())
    # return UserResponse.from_orm(saved_db_user) # Or construct manually
    # For simplicity, assuming factory returns a Pydantic model directly or can be converted
    # This part needs careful mapping between domain objects and DB storage/Pydantic models
    print(f"User created by factory: {domain_user.name}") # domain_user is the object from factory
    # ... logic to save to DB and return response ...
    # This is a placeholder, actual saving and response model creation would be here.
    # For instance, the factory might return a Pydantic model directly.
    # Or you might convert the domain object from the factory to a Pydantic model for saving.
    # Example:
    # user_to_save = UserInDB(**domain_user.model_dump()) # If factory returns Pydantic
    # db_user = await db.create_user_in_db(user_to_save)
    # return UserResponse(**db_user.model_dump())

    # Simplified response for now
    return UserResponse(id=str(uuid.uuid4()), name=domain_user.name, email=domain_user.email, user_type=user_data.type)
6. API and Database Setup6.1. Stripe & Austpost APIs (Mock Implementation)The mock Python classes StripeService and AustpostService are provided in section 4.5. They do not make external calls.6.2. MongoDB Setup (Local) & MongoDBServiceInstallation: Install MongoDB Community Server from mongodb.com. Ensure it's running.Python Driver: pip install pymongo python-dotenv pydantic-settingsMongoDBService: The class structure is provided in section 4.5. It uses pymongo.Connection details are read from .env via app.core.config.settings.It provides methods for CRUD operations on collections like products, orders, purchase_records.Object-to-document mapping needs to be handled (Pydantic's .model_dump() can convert models to dicts for MongoDB, and models can be instantiated from dicts retrieved from MongoDB).7. Bootstrap Process (app/main.py)This is the single class/module for bootstrapping.from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.config import settings
from app.services.mongodb_service import mongodb_service # Import the global instance
from app.routers import products_router, cart_router, orders_router, users_router # Import your routers

# Lifespan manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to MongoDB
    mongodb_service.connect_to_database()

    # Optional: Initialize some mock data if DB is empty
    # This is a good place for it, as it runs once on startup
    if mongodb_service.get_db().products.count_documents({}) == 0:
        print("Database is empty. Initializing mock products...")
        from app.models.product_models import ProductCreate # Import here to avoid circular deps
        sample_products = [
            ProductCreate(name="Laptop Pro", description="High-end laptop", price=2500.00, category="Electronics", stock_quantity=10),
            ProductCreate(name="Wireless Mouse", description="Ergonomic wireless mouse", price=75.00, category="Accessories", stock_quantity=50),
            ProductCreate(name="Mechanical Keyboard", description="RGB Mechanical Keyboard", price=150.00, category="Accessories", stock_quantity=30),
        ]
        for prod_data in sample_products:
            mongodb_service.create_product(prod_data) # create_product is in MongoDBService
        print(f"{len(sample_products)} mock products initialized.")

    yield # Application runs here

    # Shutdown: Close MongoDB connection
    mongodb_service.close_database_connection()

app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan # Use the lifespan manager
)

# Include routers
app.include_router(products_router.router)
app.include_router(cart_router.router) # You'll need to create this
app.include_router(orders_router.router)
app.include_router(users_router.router) # You'll need to create this

@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.APP_NAME}"}

# To run the app (from project root): uvicorn app.main:app --reload
Explanation of app/main.py:lifespan function: FastAPI's way to handle startup and shutdown events.On startup (yield is before this): Calls mongodb_service.connect_to_database(). Optionally, initializes mock product data if the products collection is empty.On shutdown (yield is after this): Calls mongodb_service.close_database_connection().FastAPI instance: Created with the lifespan manager.Routers: Routers for different parts of the API are included.Root endpoint: A simple welcome message.8. Running the ApplicationCreate .env file in the project root with your MONGODB_URL and MONGODB_DB_NAME.Install dependencies:pip install fastapi uvicorn pymongo pydantic pydantic-settings python-dotenv
Save these to requirements.txt: pip freeze > requirements.txtRun MongoDB: Ensure your local MongoDB server is running.Start FastAPI server: From the project root directory:uvicorn app.main:app --reload
app.main:app points to the app instance in app/main.py.--reload enables auto-reloading on code changes during development.Access API: Open your browser or API client (like Postman) to http://127.0.0.1:8000.FastAPI automatically provides interactive API documentation at http://127.0.0.1:8000/docs and http://127.0.0.1:8000/redoc.9. TestingPytest: Use pytest for writing and running tests.Unit Tests: Test individual functions, methods in services (with mocked dependencies), domain logic, factories, strategies.Integration Tests: Test API endpoints using FastAPI's TestClient. This allows you to make requests to your application in memory without needing a running server. You can mock external services at this level too.This guide provides a comprehensive plan for your Python/