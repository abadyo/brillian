import unittest


class UserAccount:
    count_user = 0
    account_type = "user"

    def __init__(self, id, name, phone, address, balance) -> None:
        """
        Initialize a new user account.
        """
        assert len(name) >= 4, "Name must be at least 4 characters long"
        assert balance >= 0, "Balance cannot be negative"

        self.id = id
        self.name = name
        self.phone = phone
        self.address = address
        self.balance = balance

        # Add new account
        UserAccount.count_user += 1

    def get_balance(self) -> float:
        """
        Return the current balance.
        """
        return self.balance

    def top_up(self, amount) -> float:
        """
        Top up the balance with the given amount.
        """
        assert amount > 0, "Top-up amount must be greater than 0"
        self.balance += amount
        return self.balance

    def buy_item(self, amount) -> str:
        """
        Deduct the given amount from the balance if sufficient funds are available.
        """
        assert amount <= self.balance, "Insufficient funds"
        self.balance -= amount
        return "Error" if amount == 0 else "Success"

    def give_star(self, store, star) -> float:
        """
        Give a star rating to the store.
        """
        store.rating = (store.rating + star) / 2
        return 0

    def __str__(self):
        return (
            f"I am {self.name} living at {self.address}. Call me at {self.phone} (str)"
        )

    def __repr__(self):
        return (
            f"I am {self.name} living at {self.address}. Call me at {self.phone} (repr)"
        )


class StoreAccount(UserAccount):
    count_store = 0
    account_type = "store"  # override

    def __init__(
        self, id, name, phone, address, balance, category, rating, items
    ) -> None:
        """
        Initialize a new store account.
        """
        StoreAccount.count_store += 1

        assert len(items) > 0, "There must be at least one item for sale in the store"

        self.category = category
        self.rating = rating
        self.items = items

        super().__init__(id, name, phone, address, balance)

    def sell_item(self, name, amount, user) -> float:
        """
        Sell an item and update the balance.
        """
        _item = list(filter(lambda x: x["name"] == name, self.items))
        if _item:
            _price = _item[0]["price"] * amount
            if _item[0]["stock"] < amount:
                print("Run out of stock")
                return 0
            if user.balance >= _price:
                self.balance += _price
                # user.balance -= price # can use this instead methode in user
                print(
                    f"You bought {_item[0]['name']} for {_item[0]['price']} each, quantity: {amount}"
                )
                # reduce stock
                self.items[self.items.index(_item[0])]["stock"] -= amount
                return _price
            else:
                print("Not enough money to buy this item")
                return 0
        else:
            print("Item not found")
            return 0

    def __eq__(self, other) -> bool:
        return self.id == other.id

    def __ge__(self, other) -> bool:
        return self.rating >= other.rating

    def __le__(self, other) -> bool:
        return self.rating <= other.rating

    def __str__(self) -> str:
        return (
            f"Store {self.name} at {self.address}. Store rating: {self.rating}/5 (str)"
        )

    def __repr__(self) -> str:
        return (
            f"Store {self.name} at {self.address}. Store rating: {self.rating}/5 (repr)"
        )


class AssertTest(unittest.TestCase):
    def test_user_account(self):
        user1 = UserAccount(
            id="1",
            name="John Fallout",
            phone="081345671936",
            balance=10_000,
            address="Jln Nusa Nungging",
        )
        # self.assertEqual(user1.get_balance(), 10_000)
        self.assertRaises(
            AssertionError,
            UserAccount,
            id="1",
            name="Joh",
            phone="081345671936",
            balance=10_000,
            address="Jln Nusa Nungging",
        )
        user1.top_up(100_000)
        self.assertEqual(user1.get_balance(), 110_000)
        self.assertRaises(AssertionError, user1.top_up, -10_000)

    def test_user_buy(self):
        user1 = UserAccount(
            id="1",
            name="John Fallout",
            phone="081345671936",
            balance=10_000,
            address="Jln Nusa Nungging",
        )
        store1 = StoreAccount(
            id="1",
            name="All Caps no Cash",
            phone="081345671936",
            address="Jln Nusa Jungkir",
            balance=2_000_000,
            category="Health",
            rating=4.5,
            items=[
                {"name": "Mask", "price": 10_000, "stock": 5},
                {
                    "name": "Hand Sanitizer",
                    "price": 50_000_000,
                    "stock": 10,
                },
            ],
        )
        self.assertEqual(store1.account_type, "store")
        user1.top_up(100_000)
        self.assertEqual(
            user1.buy_item(store1.sell_item("Mask", 20, user1)),
            "Error",
        )
        self.assertEqual(
            user1.buy_item(store1.sell_item("Masker", 20, user1)),
            "Error",
        )
        self.assertEqual(
            user1.buy_item(store1.sell_item("Hand Sanitizer", 2, user1)),
            "Error",
        )
        self.assertEqual(user1.buy_item(store1.sell_item("Mask", 2, user1)), "Success")
        self.assertEqual(user1.get_balance(), 90_000)
        self.assertEqual(store1.balance, 2_020_000)
        self.assertEqual(store1.items[0]["stock"], 3)
        user1.give_star(store1, 5)
        self.assertEqual(store1.rating, 4.75)


if __name__ == "__main__":
    unittest.main()  # uncomment this to test unittest

    # user1 = UserAccount(
    #     id="1",
    #     name="John Fallout",
    #     phone="081345671936",
    #     balance=10_000,
    #     address="Jln Nusa Nungging",
    # )
    # store1 = StoreAccount(
    #     id="1",
    #     name="All Caps no Cash",
    #     phone="081345671936",
    #     address="Jln Nusa Jungkir",
    #     balance=2_000_000,
    #     category="Health",
    #     rating=4.5,
    #     items=[
    #         {"name": "Mask", "price": 10_000, "stock": 5},
    #         {
    #             "name": "Hand Sanitizer made out of pure alcohol from heaven",
    #             "price": 50_000_000,
    #             "stock": 10,
    #         },
    #     ],
    # )

    # test 1, normal scenario
    # print(f"store1 money: {store1.balance}")
    # print(f"store1 rating: {store1.rating}")
    # print("-" * 40)

    # print(f"Money: {user1.get_balance()}")
    # print("Add 100.000 to balance...")
    # user1.top_up(100_000)
    # print(f"Money: {user1.get_balance()}")
    # print("Buy 2 mask")
    # user1.buy_item(store1.sell_item("Mask", 2, user1))
    # print(f"Money: {user1.get_balance()}")
    # print("Rate the store 5 out of 5")
    # user1.give_star(store1, 5)
    # print(f"Store rating: {store1.rating}/5")
    # print("-" * 40)
    # print(f"store1 money: {store1.balance}")
    # print(f"store1 rating: {store1.rating}")

    # test 2, topup negative num
    # user1.top_up(-10_000)

    # test 3 buying item that doesnt exist
    # user1.buy_item(store1.sell_item("Bodywash", 2, user1))

    # test 4 buying item but balance not enough
    # user1.buy_item(
    #     store1.sell_item(
    #         "Hand Sanitizer made out of pure alcohol from heaven", 1, user1
    #     )
    # )

# store2 = StoreAccount(
#     id="2",
#     name="All Caps some Cash",
#     phone="081345671936",
#     address="Jln Nusa Jungkir Balik",
#     balance=200,
#     category="Pain",
#     rating=1.35,
#     items=[
#         {"name": "Chainsaw", "price": 20_000, "stock": 5},
#         {"name": "Katana", "price": 50_000, "stock": 10},
#     ],
# )
