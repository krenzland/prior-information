#!/usr/bin/env python

import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, backref
from sqlalchemy import create_engine

Base = declarative_base()

class Experiment(Base):
    __tablename__ = 'experiments'
    id = Column(Integer, primary_key=True)
    dataset = Column(String(250), nullable=False)
    train_mse = Column(Float, nullable=False)
    test_mse = Column(Float, nullable=False)
    train_r2 = Column(Float, nullable=True)
    test_r2 = Column(Float, nullable=True)

class Result(Base):
    __tablename__ = 'results'
    id = Column(Integer, primary_key=True)
    level = Column(String(250), nullable=False)
    T = Column(Integer, nullable=False)
    regularization_lambda = Column(Float, nullable=False)
    regularization_type = Column(Integer, nullable=False)
    validation_mse = Column(Float, nullable=False)
    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    experiment = relationship(Experiment, backref=backref('results', uselist=True))

def make_session():
    engine = create_engine('sqlite:///results.db')
    Base.metadata.create_all(engine)
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    return session
