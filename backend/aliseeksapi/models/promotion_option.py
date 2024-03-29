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


class PromotionOption(object):
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
        'max_amount': 'Amount',
        'min_amount': 'Amount',
        'discount': 'float',
        'time_left': 'TimeDuration',
        'stock': 'int'
    }

    attribute_map = {
        'max_amount': 'maxAmount',
        'min_amount': 'minAmount',
        'discount': 'discount',
        'time_left': 'timeLeft',
        'stock': 'stock'
    }

    def __init__(self, max_amount=None, min_amount=None, discount=None, time_left=None, stock=None):  # noqa: E501
        """PromotionOption - a model defined in OpenAPI"""  # noqa: E501

        self._max_amount = None
        self._min_amount = None
        self._discount = None
        self._time_left = None
        self._stock = None
        self.discriminator = None

        if max_amount is not None:
            self.max_amount = max_amount
        if min_amount is not None:
            self.min_amount = min_amount
        if discount is not None:
            self.discount = discount
        if time_left is not None:
            self.time_left = time_left
        if stock is not None:
            self.stock = stock

    @property
    def max_amount(self):
        """Gets the max_amount of this PromotionOption.  # noqa: E501


        :return: The max_amount of this PromotionOption.  # noqa: E501
        :rtype: Amount
        """
        return self._max_amount

    @max_amount.setter
    def max_amount(self, max_amount):
        """Sets the max_amount of this PromotionOption.


        :param max_amount: The max_amount of this PromotionOption.  # noqa: E501
        :type: Amount
        """

        self._max_amount = max_amount

    @property
    def min_amount(self):
        """Gets the min_amount of this PromotionOption.  # noqa: E501


        :return: The min_amount of this PromotionOption.  # noqa: E501
        :rtype: Amount
        """
        return self._min_amount

    @min_amount.setter
    def min_amount(self, min_amount):
        """Sets the min_amount of this PromotionOption.


        :param min_amount: The min_amount of this PromotionOption.  # noqa: E501
        :type: Amount
        """

        self._min_amount = min_amount

    @property
    def discount(self):
        """Gets the discount of this PromotionOption.  # noqa: E501

        The amount of the discount   # noqa: E501

        :return: The discount of this PromotionOption.  # noqa: E501
        :rtype: float
        """
        return self._discount

    @discount.setter
    def discount(self, discount):
        """Sets the discount of this PromotionOption.

        The amount of the discount   # noqa: E501

        :param discount: The discount of this PromotionOption.  # noqa: E501
        :type: float
        """

        self._discount = discount

    @property
    def time_left(self):
        """Gets the time_left of this PromotionOption.  # noqa: E501


        :return: The time_left of this PromotionOption.  # noqa: E501
        :rtype: TimeDuration
        """
        return self._time_left

    @time_left.setter
    def time_left(self, time_left):
        """Sets the time_left of this PromotionOption.


        :param time_left: The time_left of this PromotionOption.  # noqa: E501
        :type: TimeDuration
        """

        self._time_left = time_left

    @property
    def stock(self):
        """Gets the stock of this PromotionOption.  # noqa: E501

        The amount of stock left on an item   # noqa: E501

        :return: The stock of this PromotionOption.  # noqa: E501
        :rtype: int
        """
        return self._stock

    @stock.setter
    def stock(self, stock):
        """Sets the stock of this PromotionOption.

        The amount of stock left on an item   # noqa: E501

        :param stock: The stock of this PromotionOption.  # noqa: E501
        :type: int
        """

        self._stock = stock

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
        if not isinstance(other, PromotionOption):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
