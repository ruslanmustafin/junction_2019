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


class ImageSearchRequest(object):
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
        'upload_key': 'str',
        'currency': 'RealtimeCurrency',
        'language': 'str',
        'ship_to_country': 'str',
        'category': 'int'
    }

    attribute_map = {
        'upload_key': 'uploadKey',
        'currency': 'currency',
        'language': 'language',
        'ship_to_country': 'shipToCountry',
        'category': 'category'
    }

    def __init__(self, upload_key=None, currency=None, language='en_US', ship_to_country=None, category=None):  # noqa: E501
        """ImageSearchRequest - a model defined in OpenAPI"""  # noqa: E501

        self._upload_key = None
        self._currency = None
        self._language = None
        self._ship_to_country = None
        self._category = None
        self.discriminator = None

        if upload_key is not None:
            self.upload_key = upload_key
        if currency is not None:
            self.currency = currency
        if language is not None:
            self.language = language
        if ship_to_country is not None:
            self.ship_to_country = ship_to_country
        if category is not None:
            self.category = category

    @property
    def upload_key(self):
        """Gets the upload_key of this ImageSearchRequest.  # noqa: E501


        :return: The upload_key of this ImageSearchRequest.  # noqa: E501
        :rtype: str
        """
        return self._upload_key

    @upload_key.setter
    def upload_key(self, upload_key):
        """Sets the upload_key of this ImageSearchRequest.


        :param upload_key: The upload_key of this ImageSearchRequest.  # noqa: E501
        :type: str
        """

        self._upload_key = upload_key

    @property
    def currency(self):
        """Gets the currency of this ImageSearchRequest.  # noqa: E501


        :return: The currency of this ImageSearchRequest.  # noqa: E501
        :rtype: RealtimeCurrency
        """
        return self._currency

    @currency.setter
    def currency(self, currency):
        """Sets the currency of this ImageSearchRequest.


        :param currency: The currency of this ImageSearchRequest.  # noqa: E501
        :type: RealtimeCurrency
        """

        self._currency = currency

    @property
    def language(self):
        """Gets the language of this ImageSearchRequest.  # noqa: E501

        AliExpress language to retrieve content in. Use locale.   # noqa: E501

        :return: The language of this ImageSearchRequest.  # noqa: E501
        :rtype: str
        """
        return self._language

    @language.setter
    def language(self, language):
        """Sets the language of this ImageSearchRequest.

        AliExpress language to retrieve content in. Use locale.   # noqa: E501

        :param language: The language of this ImageSearchRequest.  # noqa: E501
        :type: str
        """

        self._language = language

    @property
    def ship_to_country(self):
        """Gets the ship_to_country of this ImageSearchRequest.  # noqa: E501

        Two character iso country code   # noqa: E501

        :return: The ship_to_country of this ImageSearchRequest.  # noqa: E501
        :rtype: str
        """
        return self._ship_to_country

    @ship_to_country.setter
    def ship_to_country(self, ship_to_country):
        """Sets the ship_to_country of this ImageSearchRequest.

        Two character iso country code   # noqa: E501

        :param ship_to_country: The ship_to_country of this ImageSearchRequest.  # noqa: E501
        :type: str
        """
        if ship_to_country is not None and len(ship_to_country) > 2:
            raise ValueError("Invalid value for `ship_to_country`, length must be less than or equal to `2`")  # noqa: E501

        self._ship_to_country = ship_to_country

    @property
    def category(self):
        """Gets the category of this ImageSearchRequest.  # noqa: E501

        AliExpress category to search in  # noqa: E501

        :return: The category of this ImageSearchRequest.  # noqa: E501
        :rtype: int
        """
        return self._category

    @category.setter
    def category(self, category):
        """Sets the category of this ImageSearchRequest.

        AliExpress category to search in  # noqa: E501

        :param category: The category of this ImageSearchRequest.  # noqa: E501
        :type: int
        """

        self._category = category

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
        if not isinstance(other, ImageSearchRequest):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
