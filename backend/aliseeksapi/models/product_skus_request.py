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


class ProductSkusRequest(object):
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
        'product_id': 'str',
        'currency': 'RealtimeCurrency',
        'language': 'str',
        'locale': 'str'
    }

    attribute_map = {
        'product_id': 'productId',
        'currency': 'currency',
        'language': 'language',
        'locale': 'locale'
    }

    def __init__(self, product_id=None, currency=None, language='en_US', locale='en_US'):  # noqa: E501
        """ProductSkusRequest - a model defined in OpenAPI"""  # noqa: E501

        self._product_id = None
        self._currency = None
        self._language = None
        self._locale = None
        self.discriminator = None

        if product_id is not None:
            self.product_id = product_id
        if currency is not None:
            self.currency = currency
        if language is not None:
            self.language = language
        if locale is not None:
            self.locale = locale

    @property
    def product_id(self):
        """Gets the product_id of this ProductSkusRequest.  # noqa: E501

        The Product ID   # noqa: E501

        :return: The product_id of this ProductSkusRequest.  # noqa: E501
        :rtype: str
        """
        return self._product_id

    @product_id.setter
    def product_id(self, product_id):
        """Sets the product_id of this ProductSkusRequest.

        The Product ID   # noqa: E501

        :param product_id: The product_id of this ProductSkusRequest.  # noqa: E501
        :type: str
        """

        self._product_id = product_id

    @property
    def currency(self):
        """Gets the currency of this ProductSkusRequest.  # noqa: E501


        :return: The currency of this ProductSkusRequest.  # noqa: E501
        :rtype: RealtimeCurrency
        """
        return self._currency

    @currency.setter
    def currency(self, currency):
        """Sets the currency of this ProductSkusRequest.


        :param currency: The currency of this ProductSkusRequest.  # noqa: E501
        :type: RealtimeCurrency
        """

        self._currency = currency

    @property
    def language(self):
        """Gets the language of this ProductSkusRequest.  # noqa: E501

        AliExpress language to retrieve content in. Use locale.   # noqa: E501

        :return: The language of this ProductSkusRequest.  # noqa: E501
        :rtype: str
        """
        return self._language

    @language.setter
    def language(self, language):
        """Sets the language of this ProductSkusRequest.

        AliExpress language to retrieve content in. Use locale.   # noqa: E501

        :param language: The language of this ProductSkusRequest.  # noqa: E501
        :type: str
        """

        self._language = language

    @property
    def locale(self):
        """Gets the locale of this ProductSkusRequest.  # noqa: E501

        AliExpress locale to use.   # noqa: E501

        :return: The locale of this ProductSkusRequest.  # noqa: E501
        :rtype: str
        """
        return self._locale

    @locale.setter
    def locale(self, locale):
        """Sets the locale of this ProductSkusRequest.

        AliExpress locale to use.   # noqa: E501

        :param locale: The locale of this ProductSkusRequest.  # noqa: E501
        :type: str
        """

        self._locale = locale

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
        if not isinstance(other, ProductSkusRequest):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
