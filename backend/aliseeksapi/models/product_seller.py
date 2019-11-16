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


class ProductSeller(object):
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
        'store_id': 'int',
        'store_url': 'str',
        'store_name': 'str',
        'seller_level': 'str',
        'positive_feedback_rate': 'float'
    }

    attribute_map = {
        'store_id': 'storeId',
        'store_url': 'storeUrl',
        'store_name': 'storeName',
        'seller_level': 'sellerLevel',
        'positive_feedback_rate': 'positiveFeedbackRate'
    }

    def __init__(self, store_id=None, store_url=None, store_name=None, seller_level=None, positive_feedback_rate=None):  # noqa: E501
        """ProductSeller - a model defined in OpenAPI"""  # noqa: E501

        self._store_id = None
        self._store_url = None
        self._store_name = None
        self._seller_level = None
        self._positive_feedback_rate = None
        self.discriminator = None

        if store_id is not None:
            self.store_id = store_id
        if store_url is not None:
            self.store_url = store_url
        if store_name is not None:
            self.store_name = store_name
        if seller_level is not None:
            self.seller_level = seller_level
        if positive_feedback_rate is not None:
            self.positive_feedback_rate = positive_feedback_rate

    @property
    def store_id(self):
        """Gets the store_id of this ProductSeller.  # noqa: E501

        The ID of the seller store   # noqa: E501

        :return: The store_id of this ProductSeller.  # noqa: E501
        :rtype: int
        """
        return self._store_id

    @store_id.setter
    def store_id(self, store_id):
        """Sets the store_id of this ProductSeller.

        The ID of the seller store   # noqa: E501

        :param store_id: The store_id of this ProductSeller.  # noqa: E501
        :type: int
        """

        self._store_id = store_id

    @property
    def store_url(self):
        """Gets the store_url of this ProductSeller.  # noqa: E501

        The URL of the seller store   # noqa: E501

        :return: The store_url of this ProductSeller.  # noqa: E501
        :rtype: str
        """
        return self._store_url

    @store_url.setter
    def store_url(self, store_url):
        """Sets the store_url of this ProductSeller.

        The URL of the seller store   # noqa: E501

        :param store_url: The store_url of this ProductSeller.  # noqa: E501
        :type: str
        """

        self._store_url = store_url

    @property
    def store_name(self):
        """Gets the store_name of this ProductSeller.  # noqa: E501

        The name of the seller store   # noqa: E501

        :return: The store_name of this ProductSeller.  # noqa: E501
        :rtype: str
        """
        return self._store_name

    @store_name.setter
    def store_name(self, store_name):
        """Sets the store_name of this ProductSeller.

        The name of the seller store   # noqa: E501

        :param store_name: The store_name of this ProductSeller.  # noqa: E501
        :type: str
        """

        self._store_name = store_name

    @property
    def seller_level(self):
        """Gets the seller_level of this ProductSeller.  # noqa: E501

        The level of the seller   # noqa: E501

        :return: The seller_level of this ProductSeller.  # noqa: E501
        :rtype: str
        """
        return self._seller_level

    @seller_level.setter
    def seller_level(self, seller_level):
        """Sets the seller_level of this ProductSeller.

        The level of the seller   # noqa: E501

        :param seller_level: The seller_level of this ProductSeller.  # noqa: E501
        :type: str
        """

        self._seller_level = seller_level

    @property
    def positive_feedback_rate(self):
        """Gets the positive_feedback_rate of this ProductSeller.  # noqa: E501

        The positive feedback rate of the seller   # noqa: E501

        :return: The positive_feedback_rate of this ProductSeller.  # noqa: E501
        :rtype: float
        """
        return self._positive_feedback_rate

    @positive_feedback_rate.setter
    def positive_feedback_rate(self, positive_feedback_rate):
        """Sets the positive_feedback_rate of this ProductSeller.

        The positive feedback rate of the seller   # noqa: E501

        :param positive_feedback_rate: The positive_feedback_rate of this ProductSeller.  # noqa: E501
        :type: float
        """

        self._positive_feedback_rate = positive_feedback_rate

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
        if not isinstance(other, ProductSeller):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other