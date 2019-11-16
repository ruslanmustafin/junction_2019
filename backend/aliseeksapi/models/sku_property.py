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


class SkuProperty(object):
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
        'property_id': 'int',
        'property_name': 'str',
        'values': 'list[SkuPropertyValue]'
    }

    attribute_map = {
        'property_id': 'propertyId',
        'property_name': 'propertyName',
        'values': 'values'
    }

    def __init__(self, property_id=None, property_name=None, values=None):  # noqa: E501
        """SkuProperty - a model defined in OpenAPI"""  # noqa: E501

        self._property_id = None
        self._property_name = None
        self._values = None
        self.discriminator = None

        if property_id is not None:
            self.property_id = property_id
        if property_name is not None:
            self.property_name = property_name
        if values is not None:
            self.values = values

    @property
    def property_id(self):
        """Gets the property_id of this SkuProperty.  # noqa: E501

        The ID of the SKU property   # noqa: E501

        :return: The property_id of this SkuProperty.  # noqa: E501
        :rtype: int
        """
        return self._property_id

    @property_id.setter
    def property_id(self, property_id):
        """Sets the property_id of this SkuProperty.

        The ID of the SKU property   # noqa: E501

        :param property_id: The property_id of this SkuProperty.  # noqa: E501
        :type: int
        """

        self._property_id = property_id

    @property
    def property_name(self):
        """Gets the property_name of this SkuProperty.  # noqa: E501

        The name of the property   # noqa: E501

        :return: The property_name of this SkuProperty.  # noqa: E501
        :rtype: str
        """
        return self._property_name

    @property_name.setter
    def property_name(self, property_name):
        """Sets the property_name of this SkuProperty.

        The name of the property   # noqa: E501

        :param property_name: The property_name of this SkuProperty.  # noqa: E501
        :type: str
        """

        self._property_name = property_name

    @property
    def values(self):
        """Gets the values of this SkuProperty.  # noqa: E501

        The values of this SKU property   # noqa: E501

        :return: The values of this SkuProperty.  # noqa: E501
        :rtype: list[SkuPropertyValue]
        """
        return self._values

    @values.setter
    def values(self, values):
        """Sets the values of this SkuProperty.

        The values of this SKU property   # noqa: E501

        :param values: The values of this SkuProperty.  # noqa: E501
        :type: list[SkuPropertyValue]
        """

        self._values = values

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
        if not isinstance(other, SkuProperty):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other