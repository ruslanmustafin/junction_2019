# coding: utf-8

"""
    Aliseeks API

    AliExpress product searching and product details retrieval API.   # noqa: E501

    OpenAPI spec version: 1.0.0
    Generated by: https://openapi-generator.tech
"""


import pprint
import re  # noqa: F401

import six


class SearchItem(object):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    """
    Attributes:
      openapi_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    openapi_types = {
        'id': 'str',
        'title': 'str',
        'category_id': 'int',
        'image_url': 'str',
        'detail_url': 'str',
        'lot_size': 'int',
        'lot_unit': 'str',
        'price': 'Amount',
        'ratings': 'float',
        'orders': 'float',
        'freight': 'SearchItemFreight',
        'seller': 'SearchItemSeller',
        'freight_types': 'list[SearchItemFreightType]'
    }

    attribute_map = {
        'id': 'id',
        'title': 'title',
        'category_id': 'categoryId',
        'image_url': 'imageUrl',
        'detail_url': 'detailUrl',
        'lot_size': 'lotSize',
        'lot_unit': 'lotUnit',
        'price': 'price',
        'ratings': 'ratings',
        'orders': 'orders',
        'freight': 'freight',
        'seller': 'seller',
        'freight_types': 'freightTypes'
    }

    def __init__(self, id=None, title=None, category_id=None, image_url=None, detail_url=None, lot_size=None, lot_unit=None, price=None, ratings=None, orders=None, freight=None, seller=None, freight_types=None):  # noqa: E501
        """SearchItem - a model defined in OpenAPI"""  # noqa: E501

        self._id = None
        self._title = None
        self._category_id = None
        self._image_url = None
        self._detail_url = None
        self._lot_size = None
        self._lot_unit = None
        self._price = None
        self._ratings = None
        self._orders = None
        self._freight = None
        self._seller = None
        self._freight_types = None
        self.discriminator = None

        if id is not None:
            self.id = id
        if title is not None:
            self.title = title
        if category_id is not None:
            self.category_id = category_id
        if image_url is not None:
            self.image_url = image_url
        if detail_url is not None:
            self.detail_url = detail_url
        if lot_size is not None:
            self.lot_size = lot_size
        if lot_unit is not None:
            self.lot_unit = lot_unit
        if price is not None:
            self.price = price
        if ratings is not None:
            self.ratings = ratings
        if orders is not None:
            self.orders = orders
        if freight is not None:
            self.freight = freight
        if seller is not None:
            self.seller = seller
        if freight_types is not None:
            self.freight_types = freight_types

    @property
    def id(self):
        """Gets the id of this SearchItem.  # noqa: E501

        AliExpress Product ID   # noqa: E501

        :return: The id of this SearchItem.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this SearchItem.

        AliExpress Product ID   # noqa: E501

        :param id: The id of this SearchItem.  # noqa: E501
        :type: str
        """

        self._id = id

    @property
    def title(self):
        """Gets the title of this SearchItem.  # noqa: E501

        The subject / title of the product   # noqa: E501

        :return: The title of this SearchItem.  # noqa: E501
        :rtype: str
        """
        return self._title

    @title.setter
    def title(self, title):
        """Sets the title of this SearchItem.

        The subject / title of the product   # noqa: E501

        :param title: The title of this SearchItem.  # noqa: E501
        :type: str
        """

        self._title = title

    @property
    def category_id(self):
        """Gets the category_id of this SearchItem.  # noqa: E501

        The category of the item   # noqa: E501

        :return: The category_id of this SearchItem.  # noqa: E501
        :rtype: int
        """
        return self._category_id

    @category_id.setter
    def category_id(self, category_id):
        """Sets the category_id of this SearchItem.

        The category of the item   # noqa: E501

        :param category_id: The category_id of this SearchItem.  # noqa: E501
        :type: int
        """

        self._category_id = category_id

    @property
    def image_url(self):
        """Gets the image_url of this SearchItem.  # noqa: E501

        Image URL for the item   # noqa: E501

        :return: The image_url of this SearchItem.  # noqa: E501
        :rtype: str
        """
        return self._image_url

    @image_url.setter
    def image_url(self, image_url):
        """Sets the image_url of this SearchItem.

        Image URL for the item   # noqa: E501

        :param image_url: The image_url of this SearchItem.  # noqa: E501
        :type: str
        """

        self._image_url = image_url

    @property
    def detail_url(self):
        """Gets the detail_url of this SearchItem.  # noqa: E501

        The detail URL of the item   # noqa: E501

        :return: The detail_url of this SearchItem.  # noqa: E501
        :rtype: str
        """
        return self._detail_url

    @detail_url.setter
    def detail_url(self, detail_url):
        """Sets the detail_url of this SearchItem.

        The detail URL of the item   # noqa: E501

        :param detail_url: The detail_url of this SearchItem.  # noqa: E501
        :type: str
        """

        self._detail_url = detail_url

    @property
    def lot_size(self):
        """Gets the lot_size of this SearchItem.  # noqa: E501

        The lot size that the item is sold in   # noqa: E501

        :return: The lot_size of this SearchItem.  # noqa: E501
        :rtype: int
        """
        return self._lot_size

    @lot_size.setter
    def lot_size(self, lot_size):
        """Sets the lot_size of this SearchItem.

        The lot size that the item is sold in   # noqa: E501

        :param lot_size: The lot_size of this SearchItem.  # noqa: E501
        :type: int
        """

        self._lot_size = lot_size

    @property
    def lot_unit(self):
        """Gets the lot_unit of this SearchItem.  # noqa: E501

        The unit when describing a lot for this item   # noqa: E501

        :return: The lot_unit of this SearchItem.  # noqa: E501
        :rtype: str
        """
        return self._lot_unit

    @lot_unit.setter
    def lot_unit(self, lot_unit):
        """Sets the lot_unit of this SearchItem.

        The unit when describing a lot for this item   # noqa: E501

        :param lot_unit: The lot_unit of this SearchItem.  # noqa: E501
        :type: str
        """

        self._lot_unit = lot_unit

    @property
    def price(self):
        """Gets the price of this SearchItem.  # noqa: E501


        :return: The price of this SearchItem.  # noqa: E501
        :rtype: Amount
        """
        return self._price

    @price.setter
    def price(self, price):
        """Sets the price of this SearchItem.


        :param price: The price of this SearchItem.  # noqa: E501
        :type: Amount
        """

        self._price = price

    @property
    def ratings(self):
        """Gets the ratings of this SearchItem.  # noqa: E501

        The ratings of this item   # noqa: E501

        :return: The ratings of this SearchItem.  # noqa: E501
        :rtype: float
        """
        return self._ratings

    @ratings.setter
    def ratings(self, ratings):
        """Sets the ratings of this SearchItem.

        The ratings of this item   # noqa: E501

        :param ratings: The ratings of this SearchItem.  # noqa: E501
        :type: float
        """

        self._ratings = ratings

    @property
    def orders(self):
        """Gets the orders of this SearchItem.  # noqa: E501

        The number of orders of this item   # noqa: E501

        :return: The orders of this SearchItem.  # noqa: E501
        :rtype: float
        """
        return self._orders

    @orders.setter
    def orders(self, orders):
        """Sets the orders of this SearchItem.

        The number of orders of this item   # noqa: E501

        :param orders: The orders of this SearchItem.  # noqa: E501
        :type: float
        """

        self._orders = orders

    @property
    def freight(self):
        """Gets the freight of this SearchItem.  # noqa: E501


        :return: The freight of this SearchItem.  # noqa: E501
        :rtype: SearchItemFreight
        """
        return self._freight

    @freight.setter
    def freight(self, freight):
        """Sets the freight of this SearchItem.


        :param freight: The freight of this SearchItem.  # noqa: E501
        :type: SearchItemFreight
        """

        self._freight = freight

    @property
    def seller(self):
        """Gets the seller of this SearchItem.  # noqa: E501


        :return: The seller of this SearchItem.  # noqa: E501
        :rtype: SearchItemSeller
        """
        return self._seller

    @seller.setter
    def seller(self, seller):
        """Sets the seller of this SearchItem.


        :param seller: The seller of this SearchItem.  # noqa: E501
        :type: SearchItemSeller
        """

        self._seller = seller

    @property
    def freight_types(self):
        """Gets the freight_types of this SearchItem.  # noqa: E501

        List of freight types available for this item   # noqa: E501

        :return: The freight_types of this SearchItem.  # noqa: E501
        :rtype: list[SearchItemFreightType]
        """
        return self._freight_types

    @freight_types.setter
    def freight_types(self, freight_types):
        """Sets the freight_types of this SearchItem.

        List of freight types available for this item   # noqa: E501

        :param freight_types: The freight_types of this SearchItem.  # noqa: E501
        :type: list[SearchItemFreightType]
        """

        self._freight_types = freight_types

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.openapi_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, SearchItem):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other