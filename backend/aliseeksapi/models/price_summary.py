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


class PriceSummary(object):
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
        'original_amount': 'PriceRange',
        'unit_original_amount': 'PriceRange',
        'discounted_amount': 'PriceRange',
        'unit_discounted_amount': 'PriceRange',
        'bulk_amount': 'PriceRange',
        'unit_bulk_amount': 'PriceRange'
    }

    attribute_map = {
        'original_amount': 'originalAmount',
        'unit_original_amount': 'unitOriginalAmount',
        'discounted_amount': 'discountedAmount',
        'unit_discounted_amount': 'unitDiscountedAmount',
        'bulk_amount': 'bulkAmount',
        'unit_bulk_amount': 'unitBulkAmount'
    }

    def __init__(self, original_amount=None, unit_original_amount=None, discounted_amount=None, unit_discounted_amount=None, bulk_amount=None, unit_bulk_amount=None):  # noqa: E501
        """PriceSummary - a model defined in OpenAPI"""  # noqa: E501

        self._original_amount = None
        self._unit_original_amount = None
        self._discounted_amount = None
        self._unit_discounted_amount = None
        self._bulk_amount = None
        self._unit_bulk_amount = None
        self.discriminator = None

        if original_amount is not None:
            self.original_amount = original_amount
        if unit_original_amount is not None:
            self.unit_original_amount = unit_original_amount
        if discounted_amount is not None:
            self.discounted_amount = discounted_amount
        if unit_discounted_amount is not None:
            self.unit_discounted_amount = unit_discounted_amount
        if bulk_amount is not None:
            self.bulk_amount = bulk_amount
        if unit_bulk_amount is not None:
            self.unit_bulk_amount = unit_bulk_amount

    @property
    def original_amount(self):
        """Gets the original_amount of this PriceSummary.  # noqa: E501


        :return: The original_amount of this PriceSummary.  # noqa: E501
        :rtype: PriceRange
        """
        return self._original_amount

    @original_amount.setter
    def original_amount(self, original_amount):
        """Sets the original_amount of this PriceSummary.


        :param original_amount: The original_amount of this PriceSummary.  # noqa: E501
        :type: PriceRange
        """

        self._original_amount = original_amount

    @property
    def unit_original_amount(self):
        """Gets the unit_original_amount of this PriceSummary.  # noqa: E501


        :return: The unit_original_amount of this PriceSummary.  # noqa: E501
        :rtype: PriceRange
        """
        return self._unit_original_amount

    @unit_original_amount.setter
    def unit_original_amount(self, unit_original_amount):
        """Sets the unit_original_amount of this PriceSummary.


        :param unit_original_amount: The unit_original_amount of this PriceSummary.  # noqa: E501
        :type: PriceRange
        """

        self._unit_original_amount = unit_original_amount

    @property
    def discounted_amount(self):
        """Gets the discounted_amount of this PriceSummary.  # noqa: E501


        :return: The discounted_amount of this PriceSummary.  # noqa: E501
        :rtype: PriceRange
        """
        return self._discounted_amount

    @discounted_amount.setter
    def discounted_amount(self, discounted_amount):
        """Sets the discounted_amount of this PriceSummary.


        :param discounted_amount: The discounted_amount of this PriceSummary.  # noqa: E501
        :type: PriceRange
        """

        self._discounted_amount = discounted_amount

    @property
    def unit_discounted_amount(self):
        """Gets the unit_discounted_amount of this PriceSummary.  # noqa: E501


        :return: The unit_discounted_amount of this PriceSummary.  # noqa: E501
        :rtype: PriceRange
        """
        return self._unit_discounted_amount

    @unit_discounted_amount.setter
    def unit_discounted_amount(self, unit_discounted_amount):
        """Sets the unit_discounted_amount of this PriceSummary.


        :param unit_discounted_amount: The unit_discounted_amount of this PriceSummary.  # noqa: E501
        :type: PriceRange
        """

        self._unit_discounted_amount = unit_discounted_amount

    @property
    def bulk_amount(self):
        """Gets the bulk_amount of this PriceSummary.  # noqa: E501


        :return: The bulk_amount of this PriceSummary.  # noqa: E501
        :rtype: PriceRange
        """
        return self._bulk_amount

    @bulk_amount.setter
    def bulk_amount(self, bulk_amount):
        """Sets the bulk_amount of this PriceSummary.


        :param bulk_amount: The bulk_amount of this PriceSummary.  # noqa: E501
        :type: PriceRange
        """

        self._bulk_amount = bulk_amount

    @property
    def unit_bulk_amount(self):
        """Gets the unit_bulk_amount of this PriceSummary.  # noqa: E501


        :return: The unit_bulk_amount of this PriceSummary.  # noqa: E501
        :rtype: PriceRange
        """
        return self._unit_bulk_amount

    @unit_bulk_amount.setter
    def unit_bulk_amount(self, unit_bulk_amount):
        """Sets the unit_bulk_amount of this PriceSummary.


        :param unit_bulk_amount: The unit_bulk_amount of this PriceSummary.  # noqa: E501
        :type: PriceRange
        """

        self._unit_bulk_amount = unit_bulk_amount

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
        if not isinstance(other, PriceSummary):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other