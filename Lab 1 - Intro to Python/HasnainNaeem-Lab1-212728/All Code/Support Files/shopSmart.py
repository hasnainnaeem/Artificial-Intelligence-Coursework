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
            print("\tSorry " + " don't have fruit: %s" % (fruit))
            return None
        return self.fruitPrices[fruit]

    def getPriceOfOrder(self, orderList):
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


def shopSmart(orderList, fruitShopList):
    """ returns FruitShop where the total order cost is minimum. """

    minimumTotalCost = fruitShopList[0].getPriceOfOrder(orderList)
    fruitShopWithMinimumCost = fruitShopList[0]
    for fruitShop in fruitShopList:
        totalCost = fruitShop.getPriceOfOrder(orderList)

        if minimumTotalCost < minimumTotalCost:
            fruitShopWithMinimumCost = fruitShop
            minimumTotalCost = totalCost

    print(fruitShopWithMinimumCost.getName() + " offers the fruits with minimum total cost of : " + str(minimumTotalCost))
    return minimumTotalCost


def __main__():
    # dics of fruit prices for different shops
    fruitPricesForShop1 = {"apple": 50, "orange": 120, "pear": 200, "banana": 100}
    fruitPricesForShop2 = {"apple": 60, "orange": 250, "pear": 200, "banana": 50}
    fruitPricesForShop3 = {"apple": 70, "orange": 110, "pear": 160, "banana": 60}

    fruitPricesList = [fruitPricesForShop1, fruitPricesForShop2, fruitPricesForShop3]

    # FruitShop objects with different fruit prices
    fruitShop1 = FruitShop("Hasnain's Fruit Shop", fruitPricesForShop1)
    fruitShop2 = FruitShop("Ahmad's Fruit Shop", fruitPricesForShop2)
    fruitShop3 = FruitShop("Akbar's Fruit Shop", fruitPricesForShop3)

    fruitShopList = [fruitShop1, fruitShop2, fruitShop3]
    orderList = [('apple', 2), ("orange", 3), ("Awesome", 2)]
    minimumOrderPrice = shopSmart(orderList, fruitShopList)

# main code begins here
__main__()