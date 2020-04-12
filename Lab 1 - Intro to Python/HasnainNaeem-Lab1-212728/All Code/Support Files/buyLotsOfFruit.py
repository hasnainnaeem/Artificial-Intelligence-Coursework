"""
Notes about task:
- Have not imported shop module. Instead, converted the given python 2 code into python 3 code and used it here.

"""


class FruitShop:
    def __init__(self, name, fruitPrices):
        """
            name: Name of the fruit shop

            fruitPrices: Dictionary with keys as fruit
            strings and prices for values e.g.
            {'apples':2.00, 'oranges': 1.50, 'pears': 1.75}
        """
        self.fruitPrices = fruitPrices
        self.name = name
        print(f'Welcome to {name} fruit shop')

    def getCostPerPound(self, fruit):
        """
            fruit: Fruit string
        Returns cost of 'fruit', assuming 'fruit'
        is in our inventory or None otherwise
        """
        if fruit not in self.fruitPrices:
            print("\tSorry we don't have fruit: %s" % (fruit))
            return None
        return self.fruitPrices[fruit]

    def buyLotsOfFruits(self, orderList):
        """
            orderList: List of (fruit, numPounds) tuples

        Returns cost of orderList. If any of the fruit are
        """

        totalCost = 0.0
        for fruit, numPounds in orderList:
            costPerPound = self.getCostPerPound(fruit)
            if costPerPound is not None:
                totalCost += numPounds * costPerPound
        return totalCost

    def getName(self):
        return self.name

    def __str__(self):
        return "<FruitShop: %s>" % self.getName()


def ___main__():
    fruitPrices = {"apple": 50, "orange": 100, "pear": 150, "banana": 200}
    orderList = [('apple', 2), ("orange", 3), ("Awesome", 2)]
    fruitShop = FruitShop("Hasnain's Fruit Shop", fruitPrices)

    priceOfOrderList = fruitShop.buyLotsOfFruits(orderList)
    print("\tTotal cost of order list: " + str(priceOfOrderList))


# actual main begins here
___main__()
