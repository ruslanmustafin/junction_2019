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


class ProductBulkOption(object):
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
        'price': 'Amount',
        'discount': 'float',
        'bulk_order_count': 'int'
    }

    attribute_map = {
        'price': 'price',
        'discount': 'discount',
        'bulk_order_count': 'bulkOrderCount'
    }

    def __init__(self, price=None, discount=None, bulk_order_count=None):  # noqa: E501
        """ProductBulkOption - a model defined in OpenAPI"""  # noqa: E501

        self._price = None
        self._discount = None
        self._bulk_order_count = None
        self.discriminator = None

        if price is not None:
            self.price = price
        if discount is not None:
            self.discount = discount
        if bulk_order_count is not None:
            self.bulk_order_count = bulk_order_count

    @property
    def price(self):
        """Gets the price of this ProductBulkOption.  # noqa: E501


        :return: The price of this ProductBulkOption.  # noqa: E501
        :rtype: Amount
        """
        return self._price

    @price.setter
    def price(self, price):
        """Sets the price of this ProductBulkOption.


        :param price: The price of this ProductBulkOption.  # noqa: E501
        :type: Amount
        """

        self._price = price

    @property
    def discount(self):
        """Gets the discount of this ProductBulkOption.  # noqa: E501

        The discount for the bulk option   # noqa: E501

        :return: The discount of this ProductBulkOption.  # noqa: E501
        :rtype: float
        """
        return self._discount

    @discount.setter
    def discount(self, discount):
        """Sets the discount of this ProductBulkOption.

        The discount for the bulk option   # noqa: E501

        :param discount: The discount of this ProductBulkOption.  # noqa: E501
        :type: float
        """

        self._discount = discount

    @property
    def bulk_order_count(self):
        """Gets the bulk_order_count of this ProductBulkOption.  # noqa: E501

        The amount to order to be considered a bulk purchase   # noqa: E501

        :return: The bulk_order_count of this ProductBulkOption.  # noqa: E501
        :rtype: int
        """
        return self._bulk_order_count

    @bulk_order_count.setter
    def bulk_order_count(self, bulk_order_count):
        """Sets the bulk_order_count of this ProductBulkOption.

        The amount to order to be considered a bulk purchase   # noqa: E501

        :param bulk_order_count: The bulk_order_count of this ProductBulkOption.  # noqa: E501
        :type: int
        """

        self._bulk_order_count = bulk_order_count

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
        if not isinstance(other, ProductBulkOption):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other